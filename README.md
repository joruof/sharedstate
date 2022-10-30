# sharedstate

Unleash python's parallelism via multiprocessing with shared application state! <br>
A fast and comfortable solution via shared memory:

## Notable Features

- application state is intuitively defined with python objects
- accessing the shared state is about as fast as accessing python variables
- support for ints, floats, strings, booleans, numpy arrays, classes of these types, and more to come
- automatic recovery from crashes (even e.g. segfaults)  
- multiprocess synchronization with granular locking down to attribute level
- automatic lock release on termination (even on e.g. segfaults)

## Limitations
- questionable code quality (hacked together in a weekend)
- bugs, probably?
- no dynamic memory management (yet?) (state structure is statically defined)
- uses linux specific features, will likely never be ported to windows
