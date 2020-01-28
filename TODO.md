# Code

- [ ] Implement Flow (Glow, WaveGlow, Real NVP, … differences?)
- [ ] Add Wave And Biases
- [ ] Refactor to add Flows
- [x] Fast-gen Wavenet code
- [ ] Implement the Logit mixture from PixelCNN++
- [x] Implement VQ-VAE
- [x] NSynth with probabilistic latent space
- [x] Generation code
- [x] Freq-plot: Fix it and hue by signal
- [x] VAE-Noise: during sample-time ⇒ only use μ (no noise)

# Other

- [x] https://github.com/francesclluis/source-separation-wavenet 
- [x] Is the latent gonna be a 2-step Markov chain
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
- [ ] _Parallel Wavenet: Fast High-Fidelity Speech Synthesis_
    - ⇒ More PixelCNN++ tricks
 
