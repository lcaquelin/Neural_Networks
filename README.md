After cloning and setting 
Do in terminal uv sync
It installs alors the packages, plus the functions
Then in a .py, on peut faire par exemple : 

from museum_guide.data import prepare_subset

style = prepare_subset("style")   # or "genre"
style["train_loader"]             # also: val_loader, test_loader, classes