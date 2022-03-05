# cactice

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [CACTICE](#cactice)
  - [Overview](#overview)
  - [Conventions](#conventions)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

`cactice` stands for **c**omputing **a**gricultural **c**rop lat**tice**s. This repository explores questions about regular spatial arrangements of plant phenotypes (e.g., in a field or greenhouse). For instance:

- How does environmental context influence morphological development?
- What mechanisms underlie spatial patterning?
- Is a given phenotype distribution highly structured or mostly random? In either case, why?
- If structure is evident, can we formalize or predict it? If so, from which (and how much) information?

**This repository is exploratory, unstable, and currently very minimal.**

## Overview

Check out the `notebooks/explore.ipynb` notebook for some examples.

## Conventions

This library makes several assumptions about datasets to which the user must conform:

- Class values are parsed as strings (and mapped internally to integers). Each distinct string is a class, regardless of numeric value: for instance, `9.5` and `9.5000` are considered distinct.