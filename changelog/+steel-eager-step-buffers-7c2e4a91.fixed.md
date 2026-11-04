Allocate the per-step dynamics buffers of `SolverSteel` at construction, so a CUDA graph that captures a new solver's first step can be released without freeing buffers that later captures use.
