use std::panic::{catch_unwind, AssertUnwindSafe};

fn check_overlap(a: usize, b: usize, bytes: usize) -> bool {
    let diff = if a > b { a - b } else { b - a };
    diff < bytes
}

unsafe fn householder_inner(
    x: *const f64, v: *const f64, out: *mut f64, dim: u64, batch: u64,
) -> i32 {
    if x.is_null() || v.is_null() || out.is_null() || dim == 0 || batch == 0 {
        return -1;
    }
    let d = dim as usize;
    let b_count = batch as usize;
    let total_bytes = match d.checked_mul(b_count).and_then(|t| t.checked_mul(8)) {
        Some(b) => b,
        None => return 3,
    };
    
    let x_a = x as usize;
    let v_a = v as usize;
    let o_a = out as usize;

    if check_overlap(x_a, o_a, total_bytes) || check_overlap(v_a, o_a, total_bytes) || check_overlap(x_a, v_a, total_bytes) {
        return 1;
    }

    let eps = std::f64::EPSILON;

    for b in 0..b_count {
        let offset = b.checked_mul(d).unwrap();
        let xb = std::slice::from_raw_parts(x.add(offset), d);
        let vb = std::slice::from_raw_parts(v.add(offset), d);
        let ob = std::slice::from_raw_parts_mut(out.add(offset), d);

        let mut v_sq = 0.0_f64;
        let mut c_v = 0.0_f64;
        let mut xv = 0.0_f64;
        let mut c_xv = 0.0_f64;

        for i in 0..d {
            let v_val = vb[i];
            let x_val = xb[i];
            
            let y_v = (v_val * v_val) - c_v;
            let t_v = v_sq + y_v;
            c_v = (t_v - v_sq) - y_v;
            v_sq = t_v;
            
            let y_xv = (x_val * v_val) - c_xv;
            let t_xv = xv + y_xv;
            c_xv = (t_xv - xv) - y_xv;
            xv = t_xv;
        }

        if v_sq < eps * 10.0 {
            ob.copy_from_slice(xb);
            continue;
        }
        
        let factor = 2.0 * xv / v_sq;
        for i in 0..d {
            ob[i] = xb[i] - factor * vb[i];
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn polydim_householder_reflect_rust(
    x: *const f64, v: *const f64, out: *mut f64, dim: u64, batch: u64,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        householder_inner(x, v, out, dim, batch)
    })) {
        Ok(r) => r,
        Err(_) => -99, 
    }
}

fn solve_4x4_2rhs(a: &mut [[f64; 4]; 4], b: &mut [[f64; 2]; 4]) -> bool {
    for col in 0..4 {
        let mut best = col;
        let mut best_v = a[col][col].abs();
        for r in (col + 1)..4 {
            let v = a[r][col].abs();
            if v > best_v { best_v = v; best = r; }
        }
        if best_v < 1e-12 { return false; }
        if best != col {
            a.swap(col, best);
            b.swap(col, best);
        }
        for r in (col + 1)..4 {
            let f = a[r][col] / a[col][col];
            for j in (col + 1)..4 { a[r][j] -= f * a[col][j]; }
            a[r][col] = 0.0;
            for j in 0..2 { b[r][j] -= f * b[col][j]; }
        }
    }
    for r in (0..4).rev() {
        for j in 0..2 {
            for c in (r + 1)..4 { b[r][j] -= a[r][c] * b[c][j]; }
            b[r][j] /= a[r][r];
        }
    }
    true
}

unsafe fn cayley_inner(
    xp: *const f64, gp: *const f64, op: *mut f64, dim: u64, alpha: f64,
) -> i32 {
    if xp.is_null() || gp.is_null() || op.is_null() || dim == 0 { return -1; }
    let d = dim as usize;
    let byte_len = match d.checked_mul(16) {
        Some(b) => b,
        None => return 3,
    };
    
    let xa = xp as usize;
    let ga = gp as usize;
    let oa = op as usize;
    if check_overlap(xa, oa, byte_len) || check_overlap(ga, oa, byte_len) || check_overlap(xa, ga, byte_len) {
        return 1;
    }

    if alpha.abs() < 1e-30 {
        if xa != oa {
            let o_sl = std::slice::from_raw_parts_mut(op, d * 2);
            let x_sl = std::slice::from_raw_parts(xp, d * 2);
            o_sl.copy_from_slice(x_sl);
        }
        return 0;
    }

    let a2 = 0.5 * alpha;
    let x_sl = std::slice::from_raw_parts(xp, d * 2);
    let g_sl = std::slice::from_raw_parts(gp, d * 2);
    let o_sl = std::slice::from_raw_parts_mut(op, d * 2);

    let mut vtu = [[0.0_f64; 4]; 4];
    let mut vtx = [[0.0_f64; 2]; 4];

    for dd in 0..d {
        let x0 = x_sl[dd * 2];
        let x1 = x_sl[dd * 2 + 1];
        let g0 = g_sl[dd * 2];
        let g1 = g_sl[dd * 2 + 1];
        let vc = [x0, x1, -g0, -g1];
        let uc = [g0, g1, x0, x1];
        for i in 0..4 {
            for j in 0..4 { vtu[i][j] += vc[i] * uc[j]; }
            vtx[i][0] += vc[i] * x0;
            vtx[i][1] += vc[i] * x1;
        }
    }

    let mut c_mat = [[0.0_f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            c_mat[i][j] = if i == j { 1.0 } else { 0.0 } + a2 * vtu[i][j];
        }
    }

    let mut rhs = [[0.0_f64; 2]; 4];
    for i in 0..4 {
        for j in 0..2 {
            let mut s = 0.0;
            for m in 0..4 { s += vtu[i][m] * vtx[m][j]; }
            rhs[i][j] = vtx[i][j] - a2 * s;
        }
    }

    if !solve_4x4_2rhs(&mut c_mat, &mut rhs) { return -2; }

    for dd in 0..d {
        let x0 = x_sl[dd * 2];
        let x1 = x_sl[dd * 2 + 1];
        let g0 = g_sl[dd * 2];
        let g1 = g_sl[dd * 2 + 1];
        let uc = [g0, g1, x0, x1];
        for j in 0..2usize {
            let mut corr = 0.0;
            for m in 0..4 { corr += uc[m] * (vtx[m][j] + rhs[m][j]); }
            o_sl[dd * 2 + j] = x_sl[dd * 2 + j] - a2 * corr;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn polydim_cayley_retract_k2_rust(
    x: *const f64, g: *const f64, out: *mut f64, dim: u64, alpha: f64,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        cayley_inner(x, g, out, dim, alpha)
    })) {
        Ok(r) => r,
        Err(_) => -99,
    }
}
