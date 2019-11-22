
# Random Graphs

Summer project with Professor Upfal on analysing random graphs.

Motivation: There are many intriguing applications of studying networks using graphs.
For example friends in a network, packets routed through a network, or twitter influencers.

## Background

Community detection (thesis_papers/community_detection_label_propagation.pdf) identifies an
algorithm for identifying communities in random graphs; Erdos-Renyi graphs. They use a
label propagation algorithm they term Max-LPA that has great convergence time and good
'quality' of communities detected.

### The algorithm and Theorems

From the paper

#### Generic LPA

Overview of algorithm
 - Initialize each node in the network with a unique label
 - On each iteration, update node labels to the label which is the most frequent in its 
   neighborhood. Ties broken randomly.
 - Max-LPA: Nodes are assigned labels uniformly at random from a large space. Ties broken
   in favor of larger label. Node includes it's own label in determining most frequent 
   label in its neighborhood.
 - Community is all nodes with the same label
 - Converged if it starts cycling through a collection of states

#### Intuition

 - A single label can quickly become the mot frequent label in neighborhoods within a dense
   cluster whereas labels have trouble 'travelling' across a sparce set of endges connecting
   two dense clusters. 

#### Analysis of algorithm

 - Poljak and Sura show Max-LPA converges to a cycle of period 1 or 2
 - Shown the number of iterations of label updates for convergence is around 5
 - LPA has properties similar to epidemic processes
 - Synchronous. ie algorithm proceeds in rounds. Each round each node sends its label to all
   neighbors. Then it updates its own label

#### Theorems

Theorem 1
When Max-LPA is executed on a path $P_n$, it converges to a state from which no further
label updates occur in $O(\log(n))$ rounds w.h.p. Furthermore, in such a state there are
$\Omega(n)$ communities.

Theorem 2
Let $G(\Pi,\pi, p')$ be a clustered Erdos-Renyi graph. Suppose that the probabilities
$\{p_i\}$ and $p_i$ and the node subset sizes $\{n_i\}$ and $n$ satisfy the inequalities.

\[
 (i) n_i p_i^2 > 8np'
\]

\[
 (ii) n_i p_i^4 > 1800c \log(n)
\]

for some constant $c$. Then, given input $G(\Pi,\pi, p')$, Max-LPA converges correctly to
node patition $\Pi$ in two rounds w.h.p. (Note that condition (ii) implies for each $i$, $p_i >
\frac{log(n_i)}{n_i}$.


## the program - mirai.py

The program has been iterated numerous times (previous code found in gomi/), ending up with a base python
program that takes numerous command line arguments for adjusting parameters to generate communities
of Erdos-Renyi graphs, running episodes for label propagation, and generating images to view them.

## Observations

- There is a gap between the minimum p-value for connectedness and the intra-community p-value required for theorem II
- Empirically it appears p can be decreased theorem II's p-value to some degree with constant p' and still satisfactorily
identify the communities
- In most graphs, the average path length is less than two, allowing with high probability the community label to reach
all nodes in its community in just two steps.
- Even if the average path length increases beyond two for a large network with more than say four communities, the
intra-community path length doesn't change; about 1.5
- We can decrease the number of neighbors that distribute their p-values to about 0.7 for each node and still identify
the communities. This is likely related to the ability to lower the p value in note above

Please view images of various communities (k = number of communities studied, succes/failure = whether algorithm converged)

Constant p values
- We see with p' 0.3 and p 0.543 at n=100 and k = 2, the algorithm fails to converge at all.
This suggests E[n_p'] < 30 may succeed.
- We see with p' 0.16 and p 0.543 at n=100 and k = 4 the algorithm starts to fail to converge.
This suggests E[n_p'] = 48 (100*0.16*3) < E[n_p] = 54
- We see with p' 0.11 and p 0.543 at n=100 and k = 6 the algorithm starts to fail to converge.
This suggests E[n_p'] = 55 (100*0.11*5) > E[n_p] = 54
- We see with p' 0.09 and p 0.543 at n=100 and k = 8 the algorithm starts to fail to converge.
This suggests E[n_p'] = 63 (100*0.16*7) > E[n_p] = 54
- This suggests the algorithm is less concerned about the total number of edges from nodes outside its community and
more concerned with the proportion of edges recieved from each community.
- Together suggests as the number of communities increases the algorithm gains robustness in ability to converge even
as the total number of edges to other communities increases beyond the intra-community edges

Background noise
- If we add background noise with some number of nodes = n and connectivity to communities q we see label propagation
may still converge. It requires less connectivity to the noise as the number of nodes in the background increases.
Suggests there is a constant around E[n_q] = 30 that is a threshold.
nq 100 k 2 converge fails at q 0.30 E[n_q] = 30 < E[n_p] = 54
nq 200 k 2 converge fails at q 0.15 E[n_q] = 30 < E[n_p] = 54
nq 300 k 2 converge fails at q 0.10 E[n_q] = 30 < E[n_p] = 54
nq 400 k 2 converge fails at q 0.06 E[n_q] = 26 < E[n_p] = 54
nq 100 k 4 converge fails at q 0.36 E[n_q] = 36 < E[n_p] = 54
nq 200 k 4 converge fails at q 0.17 E[n_q] = 34 < E[n_p] = 54
nq 400 k 4 converge fails at q 0.08 E[n_q] = 32 < E[n_p] = 54
- Also increasing the number of communities gives algorithm a higher threshold in ability to converge

Variable p values
- We can allow p-values in a community to be randomized with some lower threshold. Once the minimum intra-community
probability is larger than background probability, the algorithm is able to converge. This suggests as long as a
community has a lower bound probability larger than background, the algorithm may be able to find the community.
