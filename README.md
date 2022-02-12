# CACTICE

A Python3 toolkit for **C**omputing **A**gricultural **C**ategorical lat**TICE**s: category-valued grids such as crop species or phenotype classes arranged in a field or greenhouse.

**This repository is experimental and under active development.**

## Overview

TODO

## Conventions

This library makes several assumptions about datasets to which the user must conform:

- Class values are parsed as strings (and mapped internally to integers). Each distinct string is a class, regardless of numeric value: for instance, `9.5` and `9.5000` are considered distinct.