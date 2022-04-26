# Installation

```sh
./install.sh
```

# Usage

```sh
dvc repro
```


# Simulation for optimal data sharing:

To run this set of simulations the following parameters should be enables:

* cache_capacity.quantity: 1000000000 (or other large number)
* max_resources: 10000000 (or other large number)
* policies: sfifo (to guarantee that all data is kept since cache is large enough)
* resources: placeholder: none (we don't necesarily want to limit resources)
* topology: none


## Test without optimal sharing

* optimal_data_sharing.mode: "no_sharing"

## Test with optimal sharing

* optimal_data_sharing.mode: "sharing"