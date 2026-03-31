Forked code to modify Attention

inference:
```
python makemore.py -i names.txt -o names --sample-only
```

train:
```
python makemore.py -i names.txt -o names
```

changes made:
- another CausalSelfAttention class but using scaled_dot_product_attention() 
- benchmark code to compare the this new class with the original code 