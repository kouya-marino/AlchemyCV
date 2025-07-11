name: Bug Report
description: File a bug report to help us improve
title: "[BUG] - "
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: "Thanks for taking the time to fill out this bug report!"
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: A clear and concise description of what the bug is.
    validations:
      required: true
  - type: textarea
    id: steps-to-reproduce
    attributes:
      label: Steps to Reproduce
      description: "e.g. 1. Open image 'x.png' 2. Apply 'Gaussian Blur' with kernel size 25..."
    validations:
      required: true
  - type: textarea
    id: expected-behavior
    attributes:
      label: Expected Behavior
      description: What did you expect to happen?
    validations:
      required: true
  - type: input
    id: python-version
    attributes:
      label: Python Version
      placeholder: "e.g. 3.10.4"
    validations:
      required: true
  - type: input
    id: os
    attributes:
      label: Operating System
      placeholder: "e.g. Windows 11, macOS Sonoma, Ubuntu 22.04"
    validations:
      required: true