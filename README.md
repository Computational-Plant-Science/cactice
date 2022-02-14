<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [CACTICE](#cactice)
  - [Overview](#overview)
  - [Conventions](#conventions)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# CACTICE

A Python3 toolkit for **C**omputing **A**gricultural **C**ategorical lat**TICE**s: category-valued grids such as crop species or phenotype classes arranged in a field or greenhouse.

**This repository is experimental and under active development.**

## Overview

TODO

## Conventions

This library makes several assumptions about datasets to which the user must conform:

- Class values are parsed as strings (and mapped internally to integers). Each distinct string is a class, regardless of numeric value: for instance, `9.5` and `9.5000` are considered distinct.