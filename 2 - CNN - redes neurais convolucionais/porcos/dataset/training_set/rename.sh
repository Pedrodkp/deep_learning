#!/bin/bash
i=1
for file in *.png; do
  mv "$file" "pig.$((i++)).png"
done