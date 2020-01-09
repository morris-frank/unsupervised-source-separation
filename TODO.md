# Code

- [x] NSynth with probabilistic latent space
- [x] Generation code
- [ ] Fast-gen code
- [ ] Implement the Logit mixture from PixelCNN++
- [ ] Implement VQ-VAE
- [x] Freq-plot: Fix it and hue by signal
- [x] VAE-Noise: during sample-time ⇒ only use μ (no noise)

# Other

- [ ] https://github.com/francesclluis/source-separation-wavenet 
- [ ] Is the latent gonna be a 2-step Markov chain
- [ ] Can we formulate the mixture of instrument weights in the decoder as a
    mutual information
- [ ] What metrics in MusDB, can we use the same in ToyData?

# Papers

- [ ] _Semi-Supervised Learning with Deep Generative Models_
    - ⇒ how to use conditionals in VAE 
- [x] _Fixing a Broken ELBO_
    - ⇒ about how to write ELBOs correctly
- [ ] _Density estimation using Real NVP_
    - ⇒ unsupervised learning
- [x] _PixelCNN++_
    - ⇒ Mixture of logits instead of μ-law
- [x] _Neural Discrete Representation Learning_
    - ⇒ VQ-VAE with Gumbel-SoftMax
- [ ] _Parallel WaveNet: Fast High-Fidelity Speech Synthesis_
    - ⇒ More PixelCNN++ tricks
 
