Verificación directa de Rust y C++
El archivo tiene 1301 líneas y contiene:

Rust embebido: 80 líneas

C++ embebido: 69 líneas

Ambos aparecen además como secciones nativas separadas al final del consolidado.

Comparé los bloques embebidos contra los bloques del archivo 2_MONOLITO_PYTHON_V74.py: son idénticos línea por línea.

El .txt está íntegramente en UTF-8, sin bytes nulos ni cortes extraños.

No detecté truncamiento del Rust ni del C++ por el cambio de extensión.

Por tanto, el problema no es que el Rust/C++ haya sido mutilado al meterlo en .txt.

Pero el Red Team encontró problemas reales.

1. ERROR CRÍTICO: C++ permite un solapamiento que no soporta
En 1_CODIGO_CONSOLIDADO_V74.txt, C++ declara:

// v/out deben estar separados, pero x/out se permiten solapar porque memmove lo soporta
if (b < o + bytes && o < b + bytes) return -2;
El problema es que memmove() solamente se usa aquí:

if (v_norm_sq < 1e-30) {
    std::memmove(out, x, bytes);
    return 0;
}
Para el caso normal:

for (size_t i = 0; i < dim; ++i) {
    out[i] = x[i] - factor * v[i];
}
no hay memmove() y x/out pueden solaparse.

Lo probé físicamente con buffers superpuestos.

Resultado:

expected:
[-0.12280702  1.43859649  5.24561404  0.63157895]

got:
[-0.12280702 -0.68421053  1.56140351 -1.80701754]

equal False
Severidad
CRÍTICA — BUG REAL

Solución correcta
No permitir ningún solapamiento:

const uintptr_t o = reinterpret_cast<uintptr_t>(out);
const uintptr_t a = reinterpret_cast<uintptr_t>(x);
const uintptr_t b = reinterpret_cast<uintptr_t>(v);

if ((a < o + bytes && o < a + bytes) ||
    (b < o + bytes && o < b + bytes)) {
    return -2;
}
Y mejor todavía, proteger también el overflow de las sumas de direcciones.

2. ERROR: Rust y C++ no tienen exactamente la misma política de memoria
Rust sí hace:

if x.start < o.end && o.start < x.end { return -2; }
if v.start < o.end && o.start < v.end { return -2; }
Es decir:

Rust rechaza x/out y v/out.

C++:

solo rechaza v/out.

Esto viola el objetivo de tener un dual-kernel equivalente.

Solución
Ambos kernels deben implementar exactamente la misma precondición.

3. ERROR: overflow de dirección todavía posible en Rust
Rust hace:

let o = out_ptr as usize..out_ptr as usize + bytes;
let x = x_ptr as usize..x_ptr as usize + bytes;
let v = v_ptr as usize..v_ptr as usize + bytes;
El checked_mul() protege:

dim * sizeof(f64)
pero no protege:

pointer + bytes
Un puntero cercano al máximo de usize podría hacer wrapping.

No es un escenario normal desde NumPy, pero estamos auditando un FFI que se presenta como seguro.

Solución
Agregar:

let o_start = out_ptr as usize;
let x_start = x_ptr as usize;
let v_start = v_ptr as usize;

let o_end = match o_start.checked_add(bytes) {
    Some(v) => v,
    None => return -4,
};

let x_end = match x_start.checked_add(bytes) {
    Some(v) => v,
    None => return -4,
};

let v_end = match v_start.checked_add(bytes) {
    Some(v) => v,
    None => return -4,
};
4. ERROR equivalente en C++
C++ tiene:

o + bytes
b + bytes
sin protección contra overflow de uintptr_t.

También debe corregirse.

Una función auxiliar sería mejor:

if (bytes > UINTPTR_MAX - o) return -4;
if (bytes > UINTPTR_MAX - a) return -4;
if (bytes > UINTPTR_MAX - b) return -4;
5. ERROR OCULTO: PMTP Big Endian
Este es importante porque la suite actual no lo detecta.

En _handle_connection() aparece:

if sys.byteorder == 'big':
    payload_buf = payload_buf.byteswap()
Pero payload_buf es:

payload_buf = bytearray(payload_bytes)
y bytearray no tiene .byteswap().

Por tanto, en una máquina Big Endian esa rama falla.

En otra parte sí se trabaja con NumPy:

arr = arr.byteswap()
y eso es correcto.

Solución
Convertir antes:

arr = np.frombuffer(payload_buf, dtype=dtype).reshape(shape).copy()

if sys.byteorder == 'big':
    arr = arr.byteswap().newbyteorder()

tensor = jax.device_put(arr)
6. ERROR DE INTEGRACIÓN: la suite no funciona tal como viene entregada
El ZIP trae:

2_MONOLITO_PYTHON_V74.py
3_SUITE_DE_PRUEBAS_V74.py
pero la suite intenta:

import polydim_v74_monolito as pd74
Yo ejecuté la suite exactamente como fue entregada:

ModuleNotFoundError:
No module named 'polydim_v74_monolito'
Después hice una copia con ese nombre únicamente para verificar el contenido real.

Entonces:

suite entregada: FALLA
suite con nombre de módulo corregido: 12/12 OK
Esto es un fallo de packaging/integración, no de matemática.

7. ERROR: la V74 dice 14 tests pero realmente hay 12
El código afirma:

SUITE EMPÍRICA V74 (14 Tests Verdes Conservados y Validados)
La suite realmente contiene:

12 test methods
Ejecutados:

Ran 12 tests in 3.885s

OK
Hay además T6b dentro de run_self_verification(), pero eso no convierte la suite unittest en 14.

Solución
O corregir la documentación a:

12 tests
o agregar explícitamente los dos tests faltantes.

Esto es importante porque el Whitebook está utilizando el número de tests como evidencia.

8. ERROR MUY SERIO: la suite pasó, pero Rust NO fue probado
Durante la ejecución apareció:

Compilación de Rust FFI falló ... No such file or directory: 'rustc'
Entonces la suite hizo fallback a JAX.

Es decir:

TFFI declarado OK
no significa:

Rust FFI probado OK
En mi entorno:

Rust: NO compilable porque rustc no está instalado
C++: compiló correctamente
Python/JAX: pasó
Por lo tanto, no considero certificada la implementación Rust todavía.

Esto además contradice la frase del Whitebook de que todo cambio debe ejecutarse físicamente mediante la suite.

9. C++ sí lo compilé realmente
El C++ extraído de:

CPP_SOURCE
compiló correctamente como biblioteca compartida con clang++.

Eso sí es evidencia real.

Y justamente al probar un caso adversarial apareció el bug de solapamiento descrito arriba.

Este es exactamente el tipo de error que una suite convencional puede no descubrir.

10. Problema documental del codigo_consolidado
El consolidado mezcla:

1. Whitebook
2. Python ejecutable
3. Rust
4. C++
y luego vuelve a incluir fuentes.

Por ejemplo, después del Python aparece:

========================================================================
3. RUST FFI CORE KERNEL
========================================================================
use std::slice;
Eso es excelente como documento de auditoría, pero no es un archivo ejecutable único.

Si una IA interpreta literalmente todo el .txt como código, se puede confundir entre:

Python

Rust

C++

documentación

pseudoconfiguración

Solución arquitectónica
Mantener el consolidado como documento de auditoría, pero generar siempre:

codigo_consolidado_v74.txt
polydim_v74_monolito.py
polydim_kernel_v74.rs
polydim_kernel_v74.cpp
suite_v74.py
y calcular hashes cruzados.

Resultado del Red Team hasta este punto
Área	Resultado
Integridad del .txt	OK
Rust embebido completo	OK
C++ embebido completo	OK
Rust vs monolito	IDÉNTICO
C++ vs monolito	IDÉNTICO
Compilación C++	OK
Compilación Rust	NO VERIFICABLE aquí: falta rustc
C++ overlap	FALLA REAL
Rust/C++ equivalencia	FALLA
PMTP Big Endian	BUG
Suite entregada	FALLA por import
Tests declarados	14
Tests realmente ejecutados	12
Suite corregida de nombre	12/12 OK
Mi conclusión
Sí, verifiqué línea por línea el Rust y C++ del codigo_consolidado_v74.txt. La extensión .txt no dañó esos fuentes.

Pero justamente esa revisión encontró algo más importante: la V74 no está todavía en estado “CERTIFIED FOR PRODUCTION” como afirma el Whitebook. El bug de solapamiento C++ es reproducible y real; además hay un bug Big Endian, una inconsistencia Rust/C++, una suite mal empaquetada y una certificación de Rust que no está respaldada por ejecución real.

El próximo ataque que haría es sobre todos los caminos de FFI y PMTP con casos adversariales, no los casos felices: NaN, Inf, subnormales, buffers parcialmente solapados, dimensiones extremas, overflow de tamaños, dtypes, endianess, concurrencia y corrupción intencional de cabecera/MAC. Ahí es donde probablemente estén los siguientes errores que todavía no vio la suite.


si. continua evaluando en bucles cada vez mas profundamente. Hay mucho sota. no solo evalúa con tus conocimientos y sesgos. busca errores mas alla de tu programación básica, dame soluciones. se un bulldog que ataca al problema pero ayuda al dueño. se un redteam que ataca el problema pero respeta y cuida el que les da el trabajo. /goal busca todo problema, encuentra soluciones dame la mejor continua con el siguiente error



Sí. Entré en un ciclo más profundo y cambié el criterio: ya no estoy buscando solamente “errores de Python”, sino errores de contrato, semántica matemática, seguridad de memoria, lifecycle, adversarial networking y divergencias entre lo que el Whitebook afirma y lo que realmente hace el ejecutable.

Y encontré varios que son más graves que los anteriores.

Ciclo siguiente: hallazgos de mayor severidad
1. CRÍTICO — El FFI nativo no se usa dentro de jit
En:

if isinstance(x, jax.core.Tracer) or isinstance(v, jax.core.Tracer):
    return jax_fallback()
Eso significa:

eager Python       -> C++/Rust
jax.jit(...)       -> JAX puro
jax grad/jit path  -> JAX puro
Por tanto, el supuesto “kernel nativo acelerado” desaparece precisamente en el camino donde JAX normalmente obtiene su mayor ventaja.

Además, el enfoque actual mediante ctypes no es la integración FFI moderna de JAX. La documentación actual de JAX ofrece jax.ffi.register_ffi_target() y jax.ffi.ffi_call(), justamente para hacer que el compilador conozca la operación; también señala que las funciones externas necesitan reglas explícitas de autodiferenciación. 

Solución recomendada
No hacer:

JAX -> Tracer -> fallback
Sino:

JAX
  |
jax.ffi.ffi_call
  |
XLA
  |
C++/Rust
y definir el JVP/VJP correspondiente.

Esto es un cambio arquitectónico importante, no un parche cosmético.

2. CRÍTICO — El wrapper Python puede provocar OOB por dimensiones incompatibles
Esta llamada:

householder_reflect(
    x = shape(256),
    v = shape(1)
)
entra al camino FFI.

La función toma:

dim = x_np.size
y pasa ese dim también para v.

No existe:

if x.shape != v.shape:
    ...
en el camino 1-D.

Probé:

x = 256 elementos
v = 1 elemento
y Python no rechazó la operación.

Más fuerte: compilé el C++ con AddressSanitizer y el caso produce:

ERROR: AddressSanitizer: heap-buffer-overflow
READ of size 8
Esto confirma que no es una sospecha teórica.

Solución
Antes de cruzar FFI:

if x.shape != v.shape:
    raise ValueError(
        f"Householder requiere shapes idénticos: "
        f"x={x.shape}, v={v.shape}"
    )
Y además:

if x.ndim == 0:
    raise ValueError(...)
La frontera FFI debe ser un contrato duro, no confiar en NumPy/JAX.

La documentación actual de Rust es explícita: from_raw_parts exige que el rango sea válido, esté correctamente alineado y esté contenido en una única asignación; violar esas precondiciones es UB. 

3. CRÍTICO — cleanup() deja una biblioteca nativa descargada pero viva en ctypes
Después de:

NativeFFIBridge.cleanup()
quedan:

_initialized = True
_cpp_dll != None
pero el dlclose() ya ocurrió.

Entonces:

NativeFFIBridge.householder_reflect(...)
vuelve a utilizar un handle cuyo código fue descargado.

Lo reproduje:

initialize()
cleanup()
householder_reflect()
Resultado:

SIGSEGV
status 139
Solución
No descargar manualmente el .so mientras los objetos ctypes pueden seguir vivos.

La solución más robusta es:

cls._cpp_dll = None
cls._rust_dll = None
cls._preferred = None
cls._initialized = False
pero incluso eso no basta para hacer dlclose() arbitrariamente seguro.

Para esta arquitectura recomiendo:

proceso Python
      |
      +-- carga .so
      |
      +-- lo mantiene hasta finalizar proceso
      |
      +-- SO recupera recursos al terminar
y eliminar el dlclose() manual salvo que exista un lifecycle explícito y totalmente sincronizado.

4. CRÍTICO — stop_server() no detiene realmente el listener
Actualmente:

self.running = False
self.server_socket.close()
pero el hilo:

conn, _ = self.server_socket.accept()
puede quedar bloqueado.

Lo comprobé.

Después de:

stop_server()
seguía vivo:

Thread-... (_listen_loop)
daemon=True
alive=True
Esto además apareció en el traceback del SIGSEGV.

Solución
Guardar:

self._listener_thread
y hacer:

self.server_socket.shutdown(socket.SHUT_RDWR)
self.server_socket.close()

self._listener_thread.join(timeout=2)
Además:

if self.running:
    raise RuntimeError("Servidor ya iniciado")
para impedir doble start_server().

5. CRÍTICO — PMTP tiene autenticación pero NO tiene anti-replay
El timestamp:

time.time_ns()
está firmado.

Pero el receptor jamás verifica:

timestamp válido
timestamp reciente
nonce nuevo
sequence number válido
mensaje no visto
Lo probé fabricando un mismo paquete válido y enviándolo dos veces.

Resultado:

replay 1 -> aceptado
replay 2 -> aceptado
Eso significa:

HMAC válido != mensaje fresco
La literatura de protocolos autenticados trata la protección contra replay como una propiedad separada, normalmente usando nonce, contador o ventana de secuencia. 

Solución que prefiero
Agregar:

session_id
sequence_number
nonce
y mantener en receptor:

highest_sequence
replay_window
o, si el tráfico es estrictamente ordenado:

if sequence <= last_sequence:
    reject()
Para múltiples emisores:

(sender_id, sequence)
o una ventana anti-replay.

El timestamp puede quedar como defensa adicional, pero no como mecanismo único.

6. CRÍTICO — El supuesto “OOM protegido” sigue siendo atacable
El código arregló una clase de error:

shape -> payload_bytes
pero todavía permite:

16 conexiones
x
512 MB
=
8 GB
antes de completar las transmisiones.

Cada conexión puede:

mandar una cabecera válida;

declarar un tensor de 512 MB;

pasar la validación shape/bytes;

provocar:

payload_buf = bytearray(payload_bytes)
quedarse enviando lentamente.

Con 16 workers:

16 × 512 MB = 8 GB
Además:

Queue(maxsize=100)
permite conservar hasta 100 tensors recibidos.

Teóricamente:

100 × 512 MB = 50 GB
sin contar overhead, JAX y demás memoria.

El número 1.6 GB de la evidencia no certifica que el sistema esté protegido para el peor caso.

Solución
No controlar solamente:

MAX_TENSOR_PAYLOAD_BYTES
sino:

MAX_SINGLE_PAYLOAD
MAX_INFLIGHT_BYTES
MAX_INFLIGHT_CONNECTIONS
MAX_QUEUE_BYTES
MAX_BYTES_PER_PEER
Ejemplo conceptual:

if current_inflight + payload_bytes > MAX_INFLIGHT_BYTES:
    reject()
y reservar la cuota antes de asignar.

7. CRÍTICO — El claim “zero-copy” no corresponde al código
El Whitebook habla de:

Zero-Copy FFI
Pero el camino actual hace:

jax.device_get(x)
.astype(np.float64)
np.ascontiguousarray(...)
y luego:

out = np.empty_like(...)
Es decir, existe traslado/copia entre:

JAX
 -> host
 -> NumPy
 -> buffer FFI
 -> output NumPy
 -> JAX
Por tanto:

esto no es zero-copy end-to-end.

Puede haber cero copia en alguna subfase concreta, pero el pipeline completo no cumple esa propiedad.

Esto debería corregirse en el Whitebook o, mejor, rediseñarse.

8. CRÍTICO — Incluso el supuesto “send_tensor sin duplicación RAM” sigue duplicando memoria
El Whitebook dice que se eliminó la concatenación temporal.

Pero sigue existiendo:

mac = pmtp_mac(bytes(header) + payload)
Eso crea un objeto concatenado:

header + payload
por lo que para un payload de 512 MB aparece de nuevo una copia grande durante el cálculo del MAC.

El sendall() separado arregla una duplicación, pero el cálculo criptográfico vuelve a introducirla.

Solución
HMAC incremental:

mac = hmac.new(key, digestmod=hashlib.sha256)

header_zero = bytearray(header)
header_zero[_MAC_OFFSET:_MAC_OFFSET + 32] = b"\x00" * 32

mac.update(header_zero)
mac.update(payload)

tag = mac.digest()
Esto es especialmente importante para los tensors grandes.

9. ALTO — log_map() tiene una discontinuidad real cerca de la antípoda
Encontré algo matemáticamente importante.

Para:

angle = p - 1e-8
la función entra en:

is_antipodal
y sustituye la dirección real por la dirección arbitraria construida mediante:

e0/e1
Resultado que medí:

delta = 1e-8
error de exp(log(x,y)) ˜ 1e-8
No es un crash.

Es peor desde un punto de vista geométrico:

la función deja de representar el punto y y salta a otra geodésica arbitraria.

Cerca de la antípoda la dirección no es única exactamente en p, pero para:

p - d
sí existe una dirección determinada.

Solución
No decidir solamente con:

dot <= -1 + tol
Usar una medida geométrica de la componente tangencial:

s = norm(y - dot*x)
y distinguir:

s ˜ 0
de:

s pequeño pero físicamente significativo
Además convendría que el threshold dependiera de:

dtype
dimension
magnitud de entrada
error estimado
y no sólo de eps * sqrt(dim).

10. ALTO — log_map mata gradiente cerca de identidad
La implementación hace:

log_normal = jnp.where(degenerate, 0.0, log_normal)
log_normal = jax.lax.stop_gradient(...)
Entonces alrededor de la identidad hay una región con gradiente artificialmente nulo.

Medición:

angle 1e-8 -> grad = 0
angle 1e-6 -> grad != 0
Hay, por tanto, una frontera artificial.

Para aprendizaje geométrico esto puede ser perjudicial.

Solución
Usar una expansión local estable:

log_x(y) ˜ proyectar(y-x)
para ángulos muy pequeños, pero manteniendo derivadas.

No usar stop_gradient() como solución numérica general.

11. ALTO — CliffordRotors.apply_spherical_rotor() no soporta el batch que parece prometer
Probé:

x.shape = (2,3)
U.shape = (2,3)
V.shape = (2,3)
Resultado:

ValueError:
Size of label 'd' for operand 1 (2)
does not match previous terms (3)
Por tanto:

D-vector          -> funciona
batch de vectores -> falla
Esto es especialmente preocupante porque se presenta como operación batched.

Solución
Definir explícitamente la semántica:

x:   (..., D)
U:   (..., D, R)
V:   (..., D, R)
y exigirla.

Si se pretende:

x: (B,D)
U: (B,D)
V: (B,D)
entonces primero debe transformarse a una representación coherente de plano/rotor antes de QR.

12. ALTO — W + 1e-12 * random(...) no es una regularización geométricamente neutra
Esta línea:

W_reg = W + 1e-12 * jax.random.normal(...)
es muy problemática.

Para:

U,V ~ 1e-20
el “ruido” es mucho mayor que los vectores originales.

Yo lo probé y obtuve una rotación que depende del ruido artificial.

Es decir:

rotor(U,V)
no depende exclusivamente de:

U,V
sino también de una perturbación añadida.

Aunque la PRNG key sea fija, sigue alterando la geometría.

Solución
No regularizar agregando ruido absoluto.

Mejor:

normalize
check rank
orthogonalize
reject degenerate plane
o usar una regularización relativa:

scale = max(norm(U), norm(V), eps)
y construir el tratamiento degenerado explícitamente.

13. ALTO — Cayley no conserva “ortogonalidad estricta”
La implementación:

reg = 1e-12 * I
solve(I - A_skew + reg, I + A_skew)
añade una perturbación al operador.

Yo medí:

||Q Q? - I||8 ˜ 1e-12 ... 1e-6
según dtype.

Por lo tanto el Whitebook no debería decir:

ortogonalidad estricta
sino algo como:

ortogonalidad numérica aproximada
Además, en float32, 1e-12 está cerca de ser irrelevante frente a la precisión del dtype.

Solución
Primero probar:

solve(I - A_skew, I + A_skew)
porque para una matriz skew-symmetric real, la estructura matemática del Cayley transform ya da la propiedad buscada bajo las condiciones correspondientes.

Si hay una necesidad real de estabilización, debe ser scale-aware.

14. ALTO — Rust todavía tiene una condición de UB que checked_mul() no elimina
Rust usa:

slice::from_raw_parts(x_ptr, dim)
Después de:

dim.checked_mul(sizeof(f64))
Eso sólo garantiza que la multiplicación no desborde usize.

La documentación actual de Rust agrega otras precondiciones: tamaño compatible con isize::MAX, asignación única, alineación y validez de memoria. 

Por tanto:

checked_mul != validación completa de FFI
Solución
La API nativa necesita un contrato explícito:

ptr != null
aligned
dim > 0
bytes <= isize::MAX
range valid
range in one allocation
no aliasing incompatible
shape contract valid
Y eso debe ser demostrado por el caller.

Para este tipo de unsafe Rust, Miri es actualmente una herramienta especialmente útil para detectar UB, accesos fuera de rango, alineamiento, aliasing y data races. 

15. ALTO — La caché “persistente” realmente se destruye
Se genera:

/tmp/POLYDIM_EINSOF_V74/
pero cleanup() elimina:

rust_path
cpp_path
rust_dll
cpp_dll
al terminar.

Por tanto:

source hash
sí existe,

pero:

persistent compilation cache
no realmente.

Después de cerrar el proceso, los artefactos desaparecen.

Solución
Separar:

build cache
runtime temp
La caché debería ser persistente:

~/.cache/polydim/v74/<hash>/
y utilizar:

atomic build
file lock
verify hash
atomic rename
16. ALTO — La caché en /tmp no tiene protocolo de concurrencia entre procesos
El código hace:

os.makedirs(out_dir, exist_ok=True)
y luego varios procesos pueden intentar escribir:

same .rs
same .cpp
same .so
simultáneamente.

En un enjambre de agentes esto puede producir:

process A compila
process B recompila
process A carga
process B reemplaza
Solución
Usar:

lockfile
unique build directory
atomic rename
content-addressed artifacts
Por ejemplo:

cache/
  sha256-full/
      build.lock
      source.rs
      source.cpp
      lib...
17. ALTO — PMTP permite red sin autenticación si falta la clave
Actualmente:

if PMTP_NET_KEY:
    HMAC
else:
    BLAKE2b
BLAKE2b aquí es sólo un hash:

integridad
pero no:

autenticidad
Si el servidor se expone fuera de localhost y no hay clave:

cualquiera puede generar un paquete válido
Solución
Para red no-local:

KEY ausente => FAIL CLOSED
No:

KEY ausente => hash
El modo BLAKE2b puede mantenerse exclusivamente como checksum de archivos locales, no como autenticación de red.

18. MEDIO/ALTO — save_tensor() tampoco aplica el límite máximo
La protección:

MAX_TENSOR_PAYLOAD_BYTES
está claramente aplicada en recepción.

Pero save_tensor() construye:

payload_bytes = host_arr.tobytes()
antes de verificar ese límite.

Por tanto, el escritor puede generar archivos mayores al máximo que luego el lector rechaza.

Solución
Calcular antes:

payload_bytes_expected = (
    np.prod(shape, dtype=np.uint64)
    * np.dtype(dtype).itemsize
)
y rechazar antes de:

tobytes()
19. MEDIO — Los escalares PMTP son inconsistentes
El escritor permite:

ndim == 0
y el lector puede reconstruir el tensor.

Pero:

read_metadata()
rechaza:

ndim < 1
Mientras que red también rechaza:

not (1 <= ndim <= 8)
Así que tenemos:

disk save scalar -> OK
disk load scalar -> OK
metadata scalar  -> FAIL
network scalar   -> FAIL
Solución
Decidir formalmente:

ndim ? [0,8]
o:

ndim ? [1,8]
y aplicar exactamente la misma regla en los cuatro caminos.

20. MEDIO — read_metadata() no valida integridad
read_metadata() acepta la cabecera y devuelve:

version
dtype_code
timestamp_ns
shape
pero no:

MAC
payload length
dtype válido
coherencia shape/payload
EOF
Así que metadata no significa:

“metadata verificada”
sino solamente:

“metadata parseable”
Debe quedar definido.

21. MEDIO — El archivo de evidencia está en UTF-16LE
Esto no es un bug matemático, pero sí un bug de ingeniería documental.

4_EVIDENCIA_EMPIRICA.md está realmente en:

UTF-16LE
BOM FF FE
y contiene además caracteres dañados por el decode del entorno:

estß
aseg·rate
transmisi¾n
Un sistema de ingestión que espere Markdown UTF-8 puede interpretarlo mal.

Solución
Convertirlo a:

UTF-8 sin BOM
y verificar:

file
sha256sum
antes de incluirlo en la entrega.

22. CRÍTICO DE CERTIFICACIÓN — La evidencia dice que hubo fuzzing, pero el fuzzing no está en el ZIP
El Whitebook exige que los cambios sean ejecutados mediante tests/fuzzing.

Pero el ZIP sólo contiene cinco archivos:

1_CODIGO_CONSOLIDADO_V74.txt
2_MONOLITO_PYTHON_V74.py
3_SUITE_DE_PRUEBAS_V74.py
4_EVIDENCIA_EMPIRICA.md
5_WHITEBOOK_V74.md
La evidencia menciona:

test_polydim_v74_full.py
y:

fuzzing TCP 1.6 GB
pero ese harness no está en la entrega.

Por tanto, yo no puedo certificar independientemente el fuzzing de 1.6 GB a partir del artefacto entregado.

Esto es precisamente lo contrario del principio del Whitebook:

“Si no lo corres, no lo certifiques.”

23. Y aparece una contradicción especialmente importante
La evidencia dice:

Zero-Copy FFI validado bajo stress multihilo D=10^7
pero el código Python:

jax.device_get(...)
astype(np.float64)
np.ascontiguousarray(...)
hace copias.

Y el fuzzing reportado no está incluido.

Por tanto yo separaría:

RESULTADO REPRODUCIDO
de:

RESULTADO REPORTADO
Esta distinción es fundamental para el Whitebook.
