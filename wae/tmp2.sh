#!/bin/bash

for s in dat/noisy_protos/qs_noisy_prototypes_sigma_0.*; do
    out="${s##*/}.yaml"  # Extract filename
    sed "s|fout|fout: $s|" noisy_prototypes_template2.yaml > noisy_prototypes2.yaml
    python main.py --configure noisy_prototypes2.yaml
done
