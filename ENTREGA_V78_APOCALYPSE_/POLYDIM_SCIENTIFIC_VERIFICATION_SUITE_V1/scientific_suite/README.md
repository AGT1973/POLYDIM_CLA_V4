# POLYDIM Scientific Verification Suite (SVS) v1

Suite de red-team científico para POLYDIM V78/V79.

Principio: un PASS demuestra una propiedad concreta; no demuestra que todo el sistema sea correcto.
Los tests no dependen de la propia implementación de POLYDIM como oráculo cuando existe una referencia matemática independiente.

## Capas

1. Geometría esférica: Exp/Log, identidad, ángulo local, antipodal, entradas inválidas, Jacobiano numérico.
2. Householder: invariancia de escala, involución, conservación de norma, degeneración y NaN/Inf.
3. QR: ortogonalidad, residual, rango, barrido de condición y detección de regularización que oculta rank-deficiency.
4. Cayley-SMW: equivalencia contra operador full, ortogonalidad Stiefel y sensibilidad a alpha.
5. FFI: ABI, contigüidad/layout, aliasing, trazabilidad de llamada nativa, compilación multiplataforma.
6. PMTP: tamaño real del header, round-trip, autenticidad, tamper, replay, límites de payload y overflow de shape.
7. Reproducibilidad: PRNG key independence y separación de tolerancias numéricas de reproducibilidad bitwise.
8. Metamorphic/property-based: relaciones invariantes que no dependen de ejemplos fijos.

## Uso

Desde la carpeta raíz de la entrega:

```bash
python -m pip install -U pytest hypothesis
pytest -q scientific_suite/tests
```

O, para ejecutar sólo la batería de alto valor:

```bash
pytest -q scientific_suite/tests -m "p0 or p1"
```

Para generar un reporte JSON:

```bash
python scientific_suite/tools/run_suite.py --project . --json scientific_suite/report.json
```

La suite busca `polydim_v78_monolito.py` y los kernels nativos. Los nombres históricos `.cpp.txt` / `.rs.txt` se aceptan para auditarlos, pero se reportan como incompatibilidad de packaging.

## Filosofía de estados

- PASS: la propiedad se cumple.
- FAIL: la propiedad se viola.
- ERROR: el test no pudo ejecutarse por una excepción inesperada.
- SKIP: dependencia o plataforma ausente; nunca cuenta como PASS.

No hay `xfail` automáticos para defectos conocidos.
