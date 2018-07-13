# LAP 

This repo contains MATLAB functions and scripts to run examples using the Linearize and Project (LAP) method. 

## References

This code implements the numerical techniques outlined in [this paper](https://arxiv.org/abs/1705.09992):

```
@article{Herring2018,
	title = {{LAP}: a Linearize and Project Method for Solving Inverse Problems with Coupled Variables},
	author = {J.~L.~Herring and J.~G.~Nagy and L.~Ruthotto},
	year = {2018},
	note={arXiv preprint arXiv:1705.09992 [math.NA]},
	url={https://arxiv.org/abs/1705.09992}
}
```

## Dependencies 

The codes and super-resolution examples rely heavily on the [FAIR toolbox](https://github.com/C4IR/FAIR.m). This toolbox is required in the MATLAB filepath to run any of the examples in this repo. 

Codes using hybrid regularization ('HyBR') with LAP or block coordinate descent require the [IRtools toolbox](https://github.com/jnagy1/IRtools) in the MATLAB filepath.

## Acknowledgements 

This work is supported by Emory's University Research Committee and National Science Foundation (NSF) awards DMS 1522760
and DMS 1522599. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
