# Time Skip Reinforcement Learning


Prior work:

https://arxiv.org/abs/1605.05365

This paper does something very similar, however their model adds the dynamic duration by adding a second version of each action with a different duration. I would add a second decision (either within the same model or with a second, parallel model) which selects the duration over which to perform the chosen action. 

ftp://ftp.cs.utexas.edu/pub/neural-nets/papers/braylan.aaai15.pdf

Explores use of very large (but static) frame-skip values and discovers that on some games they deliver very good results.

https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
Explanation of the motivation and mechanism behind skipping frames