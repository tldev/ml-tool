# ML Tool

## Instructions
- Pull down repo
- Add `test_data.csv` and `train_data.csv` to project folder

### Generate model:
```
python main.py train train_data.csv model_name
```

### Run tests
```
python main.py test model_name test_data.csv
```

### Generate decision tree
```
python main.py tree test_data.csv model_name completed,pns output
```