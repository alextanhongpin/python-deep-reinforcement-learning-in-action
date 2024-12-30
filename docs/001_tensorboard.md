```python
%load_ext tensorboard
```


```python
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
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


    Reusing TensorBoard on port 6006 (pid 81124), started 0:00:17 ago. (Use '!kill 81124' to kill it.)




<iframe id="tensorboard-frame-9caf145a7b1995c5" width="100%" height="800" frameborder="0">
</iframe>
<script>
  (function() {
    const frame = document.getElementById("tensorboard-frame-9caf145a7b1995c5");
    const url = new URL("/", window.location);
    const port = 6006;
    if (port) {
      url.port = port;
    }
    frame.src = url;
  })();
</script>




```python

```
