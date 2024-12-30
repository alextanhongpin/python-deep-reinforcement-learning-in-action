```python
%load_ext tensorboard
```

    The tensorboard extension is already loaded. To reload it, use:
      %reload_ext tensorboard



```python
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
# writer.close()
```


```python
import random

for epoch in range(10):
    loss = random.random()
    writer.add_scalar("Loss/train", loss, epoch)
```


```python
%tensorboard --logdir logs
```



<iframe id="tensorboard-frame-9110e9a175a267ac" width="100%" height="800" frameborder="0">
</iframe>
<script>
  (function() {
    const frame = document.getElementById("tensorboard-frame-9110e9a175a267ac");
    const url = new URL("/", window.location);
    const port = 6006;
    if (port) {
      url.port = port;
    }
    frame.src = url;
  })();
</script>




```python
for i in range(5):
    writer.add_hparams(
        {
            "num": i,
            "str": True,
            "none": None,
            # "list": [1, 2, 3],
            # "tuple": (45, 100),
            # "dict": {"one": 1, "two": 2},
        },
        {"hparam/accuracy": i, "hparam/loss": i},
        run_name="hello",
        global_step=i,
    )
```


```python
for i in range(20):
    x = random.random()
    writer.add_histogram("distribution centers", x + i, i)
```


```python
import numpy as np

for step in range(5):
    writer.add_histogram("activation", np.arange(10), step)
```


```python
writer.close()
```
