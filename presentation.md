<!-- $theme: gaia -->

# Whale Vocals Analysis

---

# Project Description

We have 10 years of deep sea audio recordings, from a hydrophone off the coast of Hawaii. This accumulates ~ `1TB` audio track per year.

The overall goal of this project is to create a vocalization/anomoly index, and eventually a classifier for events, like whale sounds.

**For this course, I am focusing on localization, and denoising.**

---

# Data

Each raw file is
- 5 minute recording (~ `14MB`)
- Single track
- Variable width peak files (custom encoding)
- 24K samples per second (96K also avalible)
- Extreamly Noisy

---

# Denoising

Denoising is a heavily studied field, and applications to underwater acoustic signal are doubly so.

There are known approaches for hydrophone denoising, but many of them require tools that were not covered in this class.

- Spectral Subtraction Method
- Emperical Noise Decomposition
- Wavelet Analysis

---

# Spectral Subtraction Method

Most familure terms to course material

- Assuming noise is stable and IID
- Fit your distribution parameters to a `pause`
- Create spectral subtraction filter
- Apply as described in class

---

# Difficulties (so far)

- Miscommunication with collaborators
  - The raw audio format is custom
  - My acceptance criteria was wrong
- Spectrograms look non-informative
- Large amounds of noise
- Clicks vs Whistles

---

# Progress

- `jupyter` dashboard for exploring the audio space
- Found several whale songs (manually)
- Design for cached audio retrieval by `datetime`

### Next 

- Implement desounding method
