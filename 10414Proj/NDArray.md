def make(
shape: 

strides: 

device: BackendDevice | None = None,

handle: Any = None,

offset: int = 0,


Create a new NDArray with the given properties. This will allocation the memory if handle=None, otherwise it will use the handle of an existing array.