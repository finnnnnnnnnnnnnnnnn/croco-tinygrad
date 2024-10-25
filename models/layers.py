class Dropout:
  def __init__(self, p=0.5):
    self.p = p

  def __call__(self, x):
    return x.dropout(p=self.p)
  
class ReLu:
  def __init__(self):
    pass

  def __call__(self, x):
    return x.relu()