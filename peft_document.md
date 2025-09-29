JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021

1

See Further for Parameter Efficient Fine-tuning by
Standing on the Shoulders of Decomposition

Chongjie Si, Xiaokang Yang, Fellow, IEEE, Wei Shen

4
2
0
2

c
e
D
5
2

]

G
L
.
s
c
[

2
v
7
1
4
5
0
.
7
0
4
2
:
v
i
X
r
a

Abstract—The rapid expansion of large foundation models
within the pre-training and fine-tuning framework has under-
scored that larger models often yield better results. However,
the scaling up of large foundation models has led to soaring
costs in fine-tuning and parameter storage, rendering exten-
sive adaptations impractical. This challenge has sparked the
development of parameter-efficient fine-tuning (PEFT), which
focuses on optimizing a select subset of parameters while keeping
the rest fixed, significantly lowering computational and storage
overheads. While recent years have witnessed a significant success
in PEFT, a deep understanding of the fundamental principles
behind these methods remains unexplored. To this end, here we
take the first step to unify all approaches by dissecting them
from a decomposition perspective. We initiate a comprehensive
mathematical analysis of these methods, allowing us to delve
deeply into their underlying mechanisms, and we explore the
reasons behind the variations in performance among different
techniques. Furthermore, inspired by our theoretical analysis,
we introduce two novel PEFT methods alongside a simple yet
effective framework designed to enhance the performance of
PEFT techniques across various applications. Our empirical
validations, conducted across multiple datasets, demonstrate the
efficacy of these methods, showcasing both theoretical validity
and practical performance improvements under the guidance
of our analytical findings. We believe our work will deepen
researchers’ understanding of PEFT and other techniques,
prompting further contemplation and advancing the research
across the whole community.

Index Terms—Parameter Efficient Fine-tuning, Decomposition

Theory, Subspace Tuning.

I. INTRODUCTION

The emergence of foundation models, as referenced in mul-
tiple studies [1]–[5], has fundamentally altered the landscape
of artificial intelligence, demonstrating substantial effective-
ness across a variety of domains. For instance, Segment Any-
thing Model (SAM) [6] has been widely implemented across
a variety of visual tasks [7]–[9], and Generative Pre-trained
Transformer (GPT) [1], [2] has even seamlessly integrated
into our daily lives, evolving into an exceedingly practical tool
[10]–[12]. Traditionally, the adaptation of pre-trained models
to specific downstream tasks required fully fine-tuning of all
parameters [13]–[15]. However, as the complexity and size of
these models have increased, this traditional approach to fine-
tuning has become less feasible, both from a computational
and resource standpoint.

In response to these challenges, there has been a pivot
towards developing more efficient techniques [16]–[19], col-

C. Si, X. Yang, and W. Shen are with MoE Key Lab of Artificial

Intelligence, AI Institute, Shanghai Jiao Tong University, Shanghai, China.

Email: {chongjiesi, xkyang, wei.shen}@sjtu.edu.cn
Codes are available at https://github.com/Chongjie-Si/Subspace-Tuning.

lectively known as parameter-efficient fine-tuning (PEFT). The
goal of PEFT is to achieve comparable or even superior per-
formance on downstream tasks by tuning a minimal number of
parameters compared with fully fine-tuning. Presently, PEFT
strategies can be categorized into three predominant groups
[20], [21], each with its distinctive mechanisms and intended
use cases.

Firstly, adapter-based methods, as discussed in several
works [18], [22]–[26], involve the insertion of small, trainable
linear modules within the pre-existing network architectures.
These modules are designed to adapt the model’s outputs
without changing the original network weights. Secondly,
the prompt-based approaches [27]–[31] make use of mutable
soft tokens placed at the beginning of inputs. This strategy
focuses on fine-tuning these prompts to steer the model’s
behavior during specific tasks. Thirdly, low-rank adaptation
approaches like LoRA derivatives [19], [20], [32]–[39] are
applied to network weights during fine-tuning, enhancing their
adaptability while maintaining overall compatibility with the
pre-trained settings. Additionally, the landscape of PEFT is
enriched by other innovative methods such as BitFit [40], [41],
which focus solely on fine-tuning the bias terms. Collectively,
these diverse strategies significantly augment the adaptability
and efficiency of models, enabling them to meet specific
task requirements without the need for extensive retraining.
Through these developments, the whole community continues
to evolve towards more sustainable and manageable model
training methodologies.

However, despite that recent years have witnessed signif-
icant advancements in PEFT [18], [42], [43], the mathemat-
ical foundations underpinning different PEFT methods have
scarcely been studied. Moreover, the performance differences
between various PEFT methods and the reasons behind these
differences have not been systematically explored. This lack
of theoretical depth limits our understanding of the potential
advantages and limitations of these methods, hindering their
optimization and innovation in practical applications. There-
fore, conducting theoretical research in this field will be crucial
for advancing PEFT technologies, providing a fundamental
basis for selecting and designing more efficient strategies.

In this paper, we undertake a pioneering theoretical exami-
nation of PEFT techniques, leveraging insights from decompo-
sition theory including matrix (decomposition) and subspace
(decomposition) theory. We introduce a novel framework
termed subspace tuning, which encapsulates all known PEFT
methods under a unified theory. The subspace tuning method
primarily focuses on adjusting the subspace of the original
parameter, involving both the reconstruction and the extension

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021

2

Fig. 1. Framework of subspace tuning. a, Subspace tuning endeavors to identify the maximal projection of the optimal weight W∗ onto the subspace spanned
by the bases of ϕ(W). Here, ϕ(W) denotes the subspace transformation of the original frozen weight W. b, Subspace reconstruction involves rescaling the
subspace of W to approximate W∗, or to construct a new subspace derived from the original. Subspace extension seeks to adjust the subspace of the original
weight W such that it approaches or even encompasses W∗. Subspace combination encompasses both the reconstruction and extension of subspaces. c, A
numerical perspective on subspace tuning. Reconstruction involves modifying the frozen parameters, while extension entails adding new tunable parameters.

of subspaces. We delve into how different methods manipulate
subspaces and elucidate the mathematical principles under-
lying each approach from the perspective of decomposition
theory. Additionally, we analyze why these methods result in
performance differences, providing a comprehensive theoret-
ical foundation to understand the dynamics within different
PEFT strategies.

Furthermore, inspired by our theoretical analysis, we pro-
pose two novel PEFT methods. Compared to existing tech-
niques, these new approaches achieve performance close to
fully fine-tuning with only 0.02% parameters. Additionally,
we introduce an effective framework that enhances the perfor-
mance of methods such as LoRA without introducing addi-
tional training parameters. This framework provides a practical
solution to optimize PEFT methodologies, thereby extending
their applicability and effectiveness in resource-constrained
environments. Extensive experiments are conducted to val-
idate our theoretical propositions by testing more than ten
methods on three different models. They not only confirm the
robustness of our theoretical insights but also demonstrate the
efficacy of the methods and framework we proposed.

We hope that our research could significantly inspire further
studies in PEFT and other related communities [25], [44]–[48],
catalyzing advancements and influencing developments across
the broader artificial intelligence landscape.

II. SUBSPACE TUNING
Consider W ∈ Rn×m as the frozen weight matrix of a layer
in a pre-trained neural network, with n ≤ m without loss of

generality. The performance of the model parameterized by
W on a specific task is quantified by a performance function
P : Rn×m → R, where a higher value indicates better
performance. Assume there exists an optimal weight matrix
W∗ ∈ Rn×m for the task at hand [21], satisfying

P(W∗) = max

¯W∈Rn×m

P( ¯W).

(1)

Typically, W∗ can be thought as the weights by fully fine-
tuning of the pre-trained model [39], [49].

The objective of PEFT methods is to approximate W∗ while
training only a small subset of parameters [39], [49]–[52].
This objective can be formalized as finding a transformation
function ϕ : Rn×m → Rn×m such that

ℓ (W∗, ϕ(W)) ,

min
ϕ

(2)

where ℓ : Rn×m × Rn×m → R is a loss function measuring
the discrepancy between W and ϕ(W), such as the Frobenius
norm:

ℓ (W∗, ϕ(W)) = ∥W∗ − ϕ(W)∥2
F .

(3)

In previous works, the function ϕ has been conceptualized
as modifications to each element of the matrix W [21]. While
this characterization is accurate, it is overly general and does
not adequately capture the underlying logic of each approach.
Indeed, from the perspective of matrix theory, adjusting W
involves modifying its associated subspaces. Therefore, we
interpret all PEFT methods as forms of Subspace Tuning (Fig.
1a). Specifically, we consider ϕ as a function that transforms
the subspaces associated with W, and Eq. (2) then aims to find

span(W)The Maximal Projection of  W*W*ϕ(W)span(ϕ(W))Subspace ViewNumerical ViewFrozen weightReconstructionExtensionWW*WW*WW*Reconstruction-basedExtension-basedCombination-basedabcSubspace TuningThe optimal weight  W*The frozen weight  WThe subspace of  span(W)span(W)Subspace of  span(W)span(W)ϕ(W)Subspace Transformation FunctionSubspace ReconstructionSubspace ExtensionSubspace CombinationW*f(W)span(f(W))WReconstructionW*span(W)span(g(W))WΔWNew Basesg(W)f(W)Subspace Reconstruction Functiong(W)Subspace Extension FunctionJOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021

3

ϕ such that ϕ(W) approximates W∗ by adjusting or extending
the subspaces of W. There are two primary strategies to
achieve this:

• Subspace Reconstruction: Modify the subspaces asso-
ciated with W through a transformation f : Rn×m →
Rn×m to better align with the subspaces of W∗. The
transformation function is then ϕ(W) = f (W).

• Subspace Extension: Introduce additional subspaces by
combining W with an additive component,
typically
represented as g : Rn×m → Rn×m, resulting in ϕ(W) =
g(W).

These strategies can be mathematically unified as

ϕ(W) = g(f (W)),

(4)

where f represents subspace reconstruction and g represents
subspace extension (Fig. 1b). Based on these strategies, we
classify existing PEFT methods into three categories:

• Reconstruction-based Methods: Adjust the subspaces of
W through transformations, aiming to align them with
those of W∗.

• Extension-based Methods: Augment W by introducing
additional components ∆W, expanding the subspace to
include new directions.

• Combination-based Methods: Integrate both reconstruc-
tion and extension strategies to adjust and expand the
subspaces simultaneously.

We will briefly introduce each category and explore the
underlying mathematical principles of corresponding methods.
The full details are left to Methods. To simplify the notations
for the following sections, let A ∈ Rn×r and B ∈ Rr×m
(r ≪ n, m) be two matrices that map the subspace to different
dimensions, with the rank being r. D ∈ Rn×m represents a
(rectangular) diagonal matrix. For a specific matrix W0 ∈
Rn×m, we use W†
0 to represent its Moore-Penrose pseudo-
inverse, and U0 ∈ Rn×n and V0 ∈ Rm×m to represent
its left and right singular vectors from its singular vector
decomposition, with Σ0 ∈ Rn×m being the corresponding
singular values. All the notations are included in Table I.

TABLE I
SUMMARY OF MAJOR NOTATIONS.

Notation
P(·)
W ∈ Rn×m
ϕ(·)
f (·)
g(·)
A ∈ Rn×r
B ∈ Rr×m
D ∈ Rn×m
W† ∈ Rm×n
U ∈ Rn×n
V ∈ Rm×m
Σ ∈ Rn×m

Mathematical Meanings
The Performance of a Model
Frozen Weight Matrix
Subspace Transformation Function
Subspace Reconstruction Function
Subspace Extension Function
Down Projection Matrix
Up Projection Matrix
(Rectangle) Diagonal Matrix
Moore-Penrose Pseudo-inverse
of the Matrix W ∈ Rn×m
Left Singular Vectors
Right Singular Vectors
Singular Values

III. SUBSPACE RECONSTRUCTION
In this section, we focus on methods that reconstruct the
subspaces associated with the weight matrix W. These meth-

ods aim to modify W through the subspace transformation,
specifically expressed as ϕ(W) = f (W), to better approxi-
mate the optimal weight matrix W∗.

We begin by considering the Singular Value Decomposition
(SVD) of W. The SVD is a fundamental technique in matrix
theory that decomposes a matrix into its constituent subspaces.
Formally, for W ∈ Rn×m, its SVD is given by:

W = UΣVT,

(5)

where U ∈ Rn×n is an orthogonal matrix whose columns
are the left singular vectors of W, forming an orthonormal
basis for the column space of W, Σ ∈ Rn×m is a diagonal
matrix with non-negative singular values σi on the diagonal,
representing the scaling factors along each principal direction,
and V ∈ Rm×m is an orthogonal matrix whose columns are
the right singular vectors of W, forming an orthonormal basis
for the row space of W. The goal of subspace reconstruction
methods is to adjust the subspaces related to U, Σ, or V to
construct a new weight matrix ϕ(W) that better approximates
W∗. We categorize these adjustments into two distinct modes
(Fig. 2a and 2b).

• Mode 1, Singular Value Adjustment: This mode entails
the modification of the singular values in Σ, thereby
adjusting the scaling within the respective principal sub-
spaces. Altering these values modifies the significance
attributed to each principal component, without affecting
the directional properties of the subspaces defined by U
and V.

• Mode 2, Singular Vector Adjustment: This mode in-
volves adjustments to the singular vectors in U and V,
including scaling the subspaces they span or more intri-
cate transformations such as reorientation or reshaping of
the subspaces. It facilitates a comprehensive adjustment
of the matrix structure.

A. Singular Value Adjustment

In Mode 1, we assume that the optimal weight matrix W
shares the same singular vectors as W. That is, we posit that
the left and right singular vectors remain unchanged, and only
the singular values need to be adjusted. Formally, we write:

W∗ = UΣ∗VT,

(6)

where Σ∗ is a diagonal matrix containing the optimal singular
values σ∗

i . The transformation function ϕ then becomes:

ϕ(W) = UΣ′VT,

(7)

where Σ′
optimize. This is what SAM-PARSER [54] actually does.

is a learnable diagonal matrix that we aim to

Specifically, SAM-PARSER targets the weight matrix of the
SAM architecture’s “neck” component, which consists of two
layers of dimension 256 × 256. The reconstruction process is
based on optimizing the 512 singular values corresponding to
these two layers. However, this method relies on the assump-
tion that the singular vectors of W and W∗ are identical,
which may not hold in practice. The limitation arises because
the adjustment only scales the principal components without
changing their directions, potentially restricting the ability to

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021

4

Fig. 2. a, Subspace view of reconstruction-based methods. Fine-tuning the singular values involves rescaling the weights, while fine-tuning the singular
vectors effectively reconstructs the subspace. b. Numerical view of reconstruction-based methods. We correspond adjustments in the subspace directly to
their numerical adjustments. c, The performance of reconstruction-based methods. With less than 0.1% of the parameters of the pretrained model, SSL and
SSB can achieve up to 99% of the performance of fully fine-tuning. The horizontal dashed line parallel to the x-axis, labeled FT, represents the performance
of fully fine-tuning. The average scores of each method are evaluated with three large pretrained models, RoBERTa-base [5], DeBERTaV3-base [53], and
RoBERTa-large [5] on the GLUE benchmark. Error bars represent the standard error of the mean across five runs.

approximate W∗ accurately. Consequently, under equivalent
conditions, the performance of the Mode 1 method is antic-
ipated to lag considerably behind that of other techniques.
Results in Fig. 2c (Supplementary Tables II-IV) proves that
this method is substantially inferior to alternative approaches,
even when allowed a larger parameter budget.

B. Singular Value Adjustment

Mode 2 methods specifically focus on the manipulation of
subspaces spanned by the singular vectors of weight matrices.
This can be formalized as follows:

ϕ(W) = T1(U)ΣT2(V)T.
(8)
where T1 : Rn×n → Rn×n and T2 : Rm×m → Rm×m, which
may be linear or nonlinear functions. It allows for scaling,
rotations, reflections, and other transformations like element-
wise nonlinear mapping of the singular vectors, providing the
greatest flexibility in adjusting the subspaces.

1) Scaling: We start from scaling the subspaces. Let D1 ∈

Rn×n and D2 ∈ Rm×m be diagonal scaling matrices.
Scale the Column Space:
If we scale the column space of
singular vectors by assigning distinct weights to each vector,
the reconstructed weight matrix ˆW can then be obtained as:
ˆW = UD1ΣD2VT = U ˆΣVT,

(9)

where ˆΣ = D1ΣD2. Therefore, scaling the column space of
singular vectors is just an adjustment of the singular values.
Scale the Row Space: We can also apply distinct weights
to each row of the singular vectors. Consequently, the recon-
structed weight matrix ˆW is articulated as follows:

ˆW = D1UΣVTD2 = D1WD2.

(10)

Thus, scaling the row space spanned by the left and right
singular vectors essentially corresponds to scaling both the
row and column spaces of the original weight matrix.

From this perspective, some methods can yield more in-
depth explanations, such as (IA)3 [55]. In the original paper
[55],
this method seeks to directly modify the activations
within the model by introducing a learnable vector l ∈ Rm
to rescale the original weight matrix. The transformation is
implemented via the Hadamard product ⊙, represented as
l ⊙ W. However, it can equivalently be expressed as WD2,
where D2 ∈ Rm×m is a diagonal matrix. Consequently, this
approach actually scales the subspace of the right singular
vectors, thereby reconstructing the original weight matrix W.
The results in Fig. 2c demonstrate that merely scaling
the subspace of the right singular vectors, i.e., (IA)3, can
achieve performance comparable to fully fine-tuning. This
insight naturally gives rise to an additional adjustment method:
Scaling the Subspace of the Left singular vectors (SSL). If

cSAM-PARSER(IA)3SSLSSBBitFit6480FT64.3487.8486.7588.1288.04Average scoreRoBERTa-largeSAM-PARSER(IA)3SSLSSBBitFit708090FT73.8786.4086.1887.4986.21Average scoreDeBERTaV3-baseRoBERTa-baseDeBERTaV3-baseRoBERTa-large0.51.0Parameter budget (‰)SAM-PARSER(IA)3SSLSSBBitFitSAM-PARSER(IA)3SSLSSBBitFit708090FT65.1383.9182.5184.7284.44Average scoreRoBERTa-baseSubspace ViewSVD U V W ΣScaleBasisBasisMode 1Fine-tune  Singular ValuesSAM-PARSER …ScaleScaleRescaleabMode 2(IA)  SSB SSL …3BitFit Preﬁx-tuning Diff pruning ….Fine-tune  Singular Vectors WSVD Σ U VSingular  Value MatrixLeft  Singular VectorsRight Singular  VectorsMode 1Fine-tune  Singular ValuesΣSAM-PARSER …Numerical ViewMode 2Fine-tune  Singular Vectors U V(IA)  SSB SSL …3BitFit Preﬁx-tuning Diff pruning …. U VJOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021

5

the dimensions of the subspaces spanned by both left and
right singular vectors are comparable, the performance of SSL
and (IA)3 are expected to be similar, since both methods
enhance model adaptation by scaling a singular subspace. This
is corroborated by the results shown in Fig. 2c.

Further expanding on this concept, we introduce the method
of Scaling the Subspace of Both left and right singular vectors
(SSB). Theoretically, SSB should outperform both SSL and
(IA)3 as it simultaneously scales both subspaces, potentially
enhancing the reconstruction quality beyond the capabilities
of single-subspace scaling. Results from Fig. 2c and detailed
in Supplementary Tables II-IV, indicate that SSB is signifi-
cantly superior to SSL and (IA)3. Additionally, while training
fewer than one-thousandth of the parameters, SSB closely
approximates the outcomes of fully fine-tuning. These findings
underscore the effectiveness of the adjustments specified in Eq.
(10), confirming the potential of subspace scaling.

2) Nonlinear Mapping: Beyond scaling, Mode 2 methods
may involve nonlinear transformations, which allows it to
capture complex relationships between W and W∗ that are
not possible with mere scaling.
BitFit and its Derivatives: BitFit [40] is designed to optimize
solely the bias terms within a model while keeping all other
parameters frozen, achieving performance comparable to full
fine-tuning. Extending this concept, S-BitFit [41] integrates
Network Architecture Search (NAS) with the BitFit strategy,
maintaining the structural
integrity of BitFit by imposing
constraints on the NAS algorithm to determine whether the
gradient of the bias term should be zero.

We consider the scenario of fine-tuning the bias term of a
layer. For an input x ∈ Rl×n with frozen weights W and bias
term b ∈ Rm, the output of the layer is computed as follows:

output = xW + 1lbT,

(11)

where 1l ∈ Rl is an all-one vector. To facilitate the integration
of the bias term into the weight matrix, we can augment W
by appending bT as an additional row. This alteration leads
to the following representation:

output = (cid:2)x 1(cid:3)

(cid:21)

(cid:20)W
bT

= ˆx ˆW,

(12)

where 1 ∈ Rl and ˆW ∈ R(n+1)×m is the augmented
matrix. Therefore, BitFit fundamentally involves fine-tuning
each element of the final row of ˆW, corresponding directly to
reconstructing the row space of the augmented weight matrix.
Soft Prompt Derivatives: Soft prompt derivatives, such as
Prefix-tuning [56], and prompt-tuning [27], are prevailing in
natural language processing [57], [58]. Prefix-tuning intro-
duces trainable continuous tokens, or prefixes, appended to
either the input or output of a layer. These prefixes, sourced
from a specific parameter matrix, remain trainable while
other parameters of the pre-trained model are fixed during
training. Conversely, Prompt-tuning simplifies this approach
by incorporating soft prompts solely at the input layer. These
prompts also originate from an independent parameter matrix
and are updated exclusively through gradient descent. Both
methods preserve the original model parameters, providing

benefits in low-data scenarios and demonstrating potential for
generalization across various tasks.

Focusing on the design rather than specific layers to place
prefixes, we consider a general case where for an input
x ∈ Rl×n and the output xW. l learnable vectors P ∈ Rl×m,
known as soft prompts, are concatenated in the following
formulation:

concat(P, xW) =

(cid:21)

(cid:20) P
xW

.

(13)

Similar to the approach used for BitFit, we can augment the

weight matrix to restate Eq. (13) as
(cid:21)

(cid:20) P
xW

=

(cid:20) I
0l×l

0l×n
x

(cid:21) (cid:20) P
W

(cid:21)

= ˆx ˆW.

(14)

Here, I ∈ Rl×l
is the identity matrix, 0l×n ∈ Rl×n and
0l×l ∈ Rl× are zero matrices, and ˆW ∈ R(n+l)×m is the
augmented matrix. Thus, soft prompt derivatives essentially
involve adjusting the elements of the initial several rows of
the augmented weight matrix ˆW, thereby reconstructing the
original subspace.
Others: There are also methods that adjust the singular vectors
by directly modifying elements within the original weight
matrix, such as Diff pruning [17], FishMask [59], Fish-Dip
[60], Xattn Tuning [61], SPT [62], and PaFi [63], etc.

IV. SUBSPACE EXTENSION

Extension-based methods aim to approximate the optimal
weight matrix W∗ by expanding the subspace spanned by
the original weight matrix W ∈ Rn×m. This is achieved by
introducing an additive component ∆W ∈ Rn×m, resulting in
an extended subspace that better captures the bases necessary
for the target task. It aims to find the closest projection of
the optimal weight W∗ within this new space (Fig. 3). The
transformation function for these methods is defined as:

ϕ(W) = g(W) = W + s∆W,

(15)

where s ∈ R is a scaling factor that adjusts the contribution of
the additive component. The goal is to find ∆W and s such
that ϕ(W) closely approximates W:

min
∆W,s

ℓ (W, W + s∆W) ,

(16)

where ℓ is a loss function, such as the Frobenius norm of the
difference.

Assuming n ≤ m,

the column space of W, denoted
is a subspace of Rn with dimension at most
as C(W),
rank(W) ≤ n. If rank(W) < n, then C(W) does not span
Rn. To approximate W∗ effectively, it may be necessary to
consider bases outside C(W). Therefore, the additive compo-
nent ∆W is expected to introduce new bases to the subspace.
Ideally, the combined column space C(W) + C(∆W) should
approximate the column space of W∗:

C(W∗) ⊆ C(W) + C(∆W).

(17)

Given that we do not have prior knowledge of C(W∗), a
conservative approach is to design ∆W such that C(W) +
C(∆W) = Rn. In the idealist scenario, the column basis

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021

6

Fig. 3. Subspace and Numerical views of extension-based methods. Extension-based methods introduce an additional weight matrix and then try to find the
optimal weight projection within the subspace spanned by this additional weight and the original weight. To achieve this, the basis of the subspace constructed
by the additional matrix should complement the basis of the original weights as much as possible. The right figure lists some common extension-based
methods and their operations on matrices.

vectors of ∆W should ideally complement
those of W,
implying that the column space of W∗ represents the direct
sum of these spaces.

Additionally, some studies [19], [39], [64] suggest that W
is highly correlated to W∗, implying that W∗ likely shares
a substantial subset of common bases with the subspace of
W. Therefore, ∆W may only need to account for a small
subset of bases absent in W but present in W∗, allowing ∆W
to be a low-rank matrix [19], [38]. Furthermore, empirical
research demonstrates that full-parameter fine-tuning of pre-
trained models can often be reparameterized into optimizations
within a low-dimensional subspace [65], [66], indicating that
the optimal weights vary within this constrained, low-rank
subspace. This low-rank characteristic of ∆W underscores
the basis for parameter efficiency in extension-based methods.
Therefore, the optimization problem then becomes:

s.t.

min
∆W

∥W∗ − W − s∆W∥2
F

rank(∆W) = r ≪ n, m.
(18)
An additional critical aspect is the scaling factor s. For fixed
W and ∆W, the optimal s∗ that minimizes the loss function
can be found by solving:

s∗ = arg min

s

∥W∗ − W − s∆W∥2
F .

(19)

This is a univariate quadratic minimization problem, solvable
in closed form:

s =

⟨W∗ − W, ∆W⟩F
∥∆W∥2
F

,

(20)

where ⟨·, ·⟩F denotes the Frobenius inner product. However,
s is usually set as a hyper-parameter [19], [37], [38]. During
training, the increment matrix ∆W changes continuously, so
different values of s can potentially guide ∆W along distinct
learning trajectories. As a result, the choice of s can have a
significant—or even critical—impact on model performance,
implicitly affecting both the scaling and the direction of
updates in ∆W.

A. LoRA and its Derivatives

Based on the hypothesis that the update ∆W is of low rank,
LoRA (Low-Rank Adaptation) introduces ∆W as a product
of two low-rank matrices:

∆W = AB,
(21)
where A ∈ Rn×r and B ∈ Rr×m, with r ≪ {n, m}. The
rank constraint ensures parameter efficiency. Without loss of
generality, assuming both A and B fully utilize the carrying
capacities of their low ranks, i.e., rank(A) = rank(B) = r.
The addition term in LoRA aligns with the forms of full rank
decomposition [67]1. Subsequent researches employ different
forms of matrix decomposition for ∆W:

• TriLoRA [70] and AdaLoRA [37] use a decomposition
incorporating a diagonal matrix D ∈ Rr×r: ∆W =
ADB.

• FLoRA [38] introduces an arbitrary matrix M ∈ Rr×r:

∆W = AMB.

We first demonstrate that different decomposition forms
are capable of being transformed into one another. Using the
general form of ∆W = AGB, we have

AMB = AUΣVB (M = UΣV)

=A⋄DB⋄ (A⋄ = AU, B⋄ = VB)
(A∗ = A⋄D).
=A∗B⋄

(22)

Nevertheless, despite their structural similarities, these config-
urations result in varying levels of effectiveness. The effective-
ness hierarchy is demonstrated as P(AMB) > P(ADB) >
P(AB) in general, as evidenced by Fig. 4 and studies such as
[19], [37], [38], [70]. Indeed, beyond the methods mentioned,
despite the potential for various methods to be theoretically
interchangeable in form, there are still performance differences
observed between them. We believe the choice of the ∆W’s
form could influence the feasible set and thus the capacity to
approximate ∆W∗ = W∗ − W, and thus we formalize this
observation with the following proposition:

1In [54], the authors assert that the formulation of LoRA is based on QR

decomposition [68], [69], which we believe is incorrect.

Subspace ViewNumerical ViewAdapter DerivativesxWoutputOutputhhhAdapterPAScaled PAORORORORscaleAAABBBLoRA DerivativesxWOROROutputORORLoRATriLoRAFLoRAFrozen  Weight  MatrixAAABBBW*WΔW1ΔW2span(W)dim(span(W*))=3dim(span(W))=2span(ΔW1)⊆span(W)span(ΔW2)⊈span(W)The Maximal Projection of W*JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021

7

Fig. 4. Average score of extension and combination-based methods. Each method is assessed under four different ranks. The horizontal dashed line parallel
to the x-axis, labeled FT, represents the performance of fully fine-tuning. In general the performance of FLoRA is superior to that of AdaLoRA and TriLoRA,
followed by LoRA, and the performance of DoRA is superior to that of LoRA. The average scores of PEFT methods are evaluated with three large pretrained
models, RoBERTa-base [5], DeBERTaV3-base [53], and RoBERTa-large [5] on the GLUE benchmark. Each method is assessed under four different ranks 2,
4, 8 and 16. Error bars represent the standard error of the mean across five runs.

Proposition 1. The expressiveness and optimization landscape
of extension-based methods are directly influenced by the de-
composition structure of ∆W, rather than by its mathematical
equivalence to other forms.

mystery in extension-based methods — beyond mere theoret-
ical equivalence and the low-rank characteristics traditionally
focused on, how exactly does each decomposition form impact
the model’s learning dynamics?

This proposition is intriguing: If various forms are in-
terchangeable and they all only apply the same low-rank
constraint on ∆W, why do we still see differences in per-
formance? Could a ∆W learned by FLoRA for a specific
weight not be equivalently learned by other methodologies
such as LoRA and TriLoRA? This paradox highlights a deeper

To delve into this issue, consider the ideal case where we
aim to match ∆W to ∆W∗: AGB = ∆W∗. We utilize SVD
on ∆W∗, resulting in G = A†UΣVT B†. Suppose A† =
PUT and B† = VΣ†Q, where P ∈ Rr×n and Q ∈ Rn×r
are arbitrary matrices, we can derive G = PQ. If G is an
identity matrix, Q must be the pseudo-inverse of P. Building
on this setup, if G is a diagonal matrix, it further specifies that

HAdapterPAdapterPALoRATriLoRAAdaLoRAFLoRADoRA88FT88.5688.7088.8088.5288.2889.0188.8588.77Average scoreRoBERTa-largeHAdapterPAdapterPALoRATriLoRAAdaLoRAFLoRADoRAFT89.0588.5588.7188.9689.0788.9589.2489.15Average scoreRoBERTa-largeHAdapterPAdapterPALoRATriLoRAAdaLoRAFLoRADoRA8687FT86.2585.7886.5486.4186.4086.2186.3786.11Average scoreRoBERTa-baseHAdapterPAdapterPALoRATriLoRAAdaLoRAFLoRADoRA888990FT88.0588.1689.0688.6589.2089.2489.2789.25Average scoreDeBERTaV3-baseHAdapterPAdapterPALoRATriLoRAAdaLoRAFLoRADoRA87888990FT87.7388.0288.9987.9588.8988.9089.3088.31Average scoreDeBERTaV3-baseHAdapterPAdapterPALoRATriLoRAAdaLoRAFLoRADoRA858687FT86.1585.6286.2086.1485.3086.2086.2186.12Average scoreRoBERTa-baseHAdapterPAdapterPALoRATriLoRAAdaLoRAFLoRADoRA8687FT86.4485.9385.8385.6485.8986.1686.1086.37Average scoreRoBERTa-baseHAdapterPAdapterPALoRATriLoRAAdaLoRAFLoRADoRA88FT88.9588.8388.5388.6888.7688.6088.8589.00Average scoreRoBERTa-largeHAdapterPAdapterPALoRATriLoRAAdaLoRAFLoRADoRA888990FT88.1988.3289.2188.1789.2089.2389.8088.49Average scoreDeBERTaV3-baseHAdapterPAdapterPALoRATriLoRAAdaLoRAFLoRADoRA8687FT85.7086.2186.0286.1285.9985.9986.3786.56Average scoreRoBERTa-baseHAdapterPAdapterPALoRATriLoRAAdaLoRAFLoRADoRAFT89.0088.9788.7788.7188.7688.6788.9588.84Average scoreRoBERTa-largeHAdapterPAdapterPALoRATriLoRAAdaLoRAFLoRADoRA888990FT88.6888.6089.4788.9189.0389.2189.3688.64Average scoreDeBERTaV3-baser=2r=4r=8r=16JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021

8

each column of Q, which is the pseudo-inverse of P, may have
distinct weights. Conversely, when G is an arbitrary matrix, it
does not impose any such constraints. Therefore, we can arrive
at the conclusion: LoRA imposes the most restrictions on the
matrix patterns for model learning, followed by AdaLoRA,
while FLoRA imposes the least.

At this point, we can explain why different decomposition
forms yield varying levels of effectiveness in model training.
Although these methods may have constraints on the matrix
patterns,
these constraints are not applied during training,
which means that LoRA and AdaLoRA are less likely to
learn the optimal change compared to FLoRA. Further, even
if specific matrix patterns are constrained during training,
FLoRA’s greater degree of freedom and stronger expressive
power in learning make it much easier to achieve better results
[71]. This leads us to propose the following:

Proposition 2. Beyond the inherent low-rank constraint of the
decomposition itself, different decomposition forms implicitly
and subtly impose additional constraints on the matrix pat-
terns. A decomposition that imposes fewer constraints allows
for a richer representation of ∆W, potentially leading to
better approximation of W∗.

B. Matrix Pattern Constraints (MPC)

The analysis of different decompositions in the LoRA
derivatives provides valuable insights into optimizing model
performance through matrix interactions. To mitigate the
limitations imposed by constrained decompositions, we can
introduce regularization terms that encourage desirable prop-
erties in A and B. Specifically, we can impose orthogonality
or diagonal constraints to enhance the expressiveness while
controlling the optimization complexity.

When G is a diagonal matrix, we simply constrain A† =
D1UT and B† = VD2, where D1 ∈ Rr×n and D2 ∈ Rm×r
are rectangular diagonal matrices. This setup inversely dictates
the formulations for A and B based on the pseudo-inverses
of D1 and D2, as A = UD†
2VT. This analysis
reveals that A and B must be structured as linear combinations
of the columns of U and V, respectively. Consequently, the
matrix relationships are captured as follows:

1 and B = D†

AAT = UD†
BTB = VD†T

1D†T
2 D†

1 UT, ATA = D†T
2VT, BBT = D†

1 D†
1,
2D†T
2 .

(23)

D†T

2D†T
and D†T

1 = D†
1D†T
1
identity matrices. This arrangement

By simplifying the constraints on D1 = Ir×n and D2 = Im×r
during training, the following identities hold:
1 D†
In this configuration, D†
2 function as block
diagonal
introduces a
degree of semi-orthogonality to both A and B, since ATA =
BBT = Ir and AAT and BTB are arbitrary matrices.
Therefore, when G is a diagonal matrix, we can impose semi-
positive definite constraints on the two matrices by introducing
regularization term:

2 = Ir.
2 D†

(24)

min ∥ATA − Ir∥2

F + ∥BBT − Ir∥2
F .

(25)

Extending this further, when G is an identity matrix, we
obtain I = A†∆W∗B† = D1ΣD2. It becomes evident that
D1 and D2 are intricately correlated. Therefore, it is feasible
to first impose a semi-positive definite constraint to at most one
of A and B, such as A. For the other matrix B, we can adopt
two types of constraints. One is also constraining B to exhibit
orthogonality, with the regularizer formulated in the Eq. (25).
This constraint assumes that Σ of ∆W∗ is an identity matrix,
which makes it a more aggressive approach. Alternatively,
we can relax the constraint. Eq. (23) suggests that BTB is
a diagonal matrix, and hence, we can impose a diagonal
constraint on B, with the corresponding regularization term
as:

min ∥ATA − I∥2

F + ∥BBT − diag(BBT)∥2
F .

(26)

that

Furthermore, we can directly break the connection between
D1 and D2,
is, between A and B,
to alleviate the
constraints during the learning process. This can be achieved
by introducing a non-linear operation between the matrix
multiplications of A and B, effectively breaking the strong
link that typically exists between them.

Overall, we unify the aforementioned three Matrix Pattern
Constraints as a novel framework termed as MPC, with
MPCo (Eq. (25)), MPCd (Eq. (26)) and MPCn (nonlin-
ear involvement). Without introducing additional parameters,
MPC effectively enhance the performance of existing LoRA
derivatives, as shown in Figs. 5a-5c (Supplementary Tables.II-
IV). Moreover, MPC can help different methods achieve more
stable training. Methods combined with MPC generally exhibit
smaller standard deviations compared to those without MPC.

C. Adapter Derivatives

Adapter derivatives [18], [22], [72], as a pioneering ap-
proach in PEFT,
integrate small neural modules to adapt
models effectively. For an input x ∈ Rn×m, the adapters are
integrated with a residual connection, resulting in the final
transformation:

x → x + h(xA)B,

(27)

where h(·) is a nonlinear activation function, A ∈ Rm×r and
B ∈ Rr×m. This configuration can be further expressed by
considering weight matrix as the input ˆW = x ∈ Rn×m and
a hypothetical input ˆx = I ∈ Rm×m as

ˆWˆx → ˆWˆx + h( ˆWA)Bˆx,

(28)

Therefore, the addition term introduced by the Adapter can be
formulated as ∆W = h( ˆWA)B, and we can derive

h( ˆWA) = ∆W∗B†.

(29)

Compared with LoRA, where A = ∆W∗B†, the Adapter’s
inclusion of a nonlinear activation layer further reduces the
constraints on learning the relationships between A and B.
According to Proposition 2,
this reduction in constraints
should lead to better model performance. However, the input to
the Adapter is fixed as I, which implies that Adapters can only
be placed at specific modules within a backbone architecture,
potentially limiting their overall effectiveness.

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021

9

Fig. 5. Average score of different methods coupled with MPC framework. a, The performance of LoRA when coupled with MPCo, MPCd, and MPCn.
b-c, The performance of TriLoRA and AdaLoRA when coupled with MPCo, respectively. The MPC framework significantly enhances the performance of
various PEFT methods, as evaluated with three large pretrained models, RoBERTa-base [5], DeBERTaV3-base [53], and RoBERTa-large [5] on the GLUE
benchmark. Each method is assessed under four different ranks 2, 4, 8 and 16. Error bars represent the standard error of the mean across five runs.

Subsequent developments, such as the MAM Adapter [18],
which includes the PA or Scaled PA, adopt a parallel architec-
ture similar to LoRA to configure adapter layers. Specifically,
for an input x ∈ Rd×n, we have

its application across various model positions. This nonlinear
flexibility typically results in enhanced model performance, as
evidenced by our experiments.

xW → xW + h(xA)B,

(30)

V. SUBSPACE COMBINATION

where A ∈ Rn×r and B ∈ Rr×m. Following Eq. (28), we
have

ˆWˆx → ˆWˆx + h( ˆWA)B,
(31)
resulting in ∆W = h( ˆWA)Bˆx†. This configuration allows
Adapters to be applied to any weight matrix within the
model. We focus particularly on the design of the PA and

Combination-based methods perform both subspace recon-
struction and extension simultaneously, blending the principles
of both approaches. Moreover, for some methods which can be
categorized as both a reconstruction-based and an extension-
based method, we also classify them as the combination-based
methods. We here analyze several representative combination-
based methods as follows.

248168788899088.2888.7688.7689.0788.8988.8488.8689.12Average scoreRoBERTa-large248168788899088.8388.4789.1688.6089.0188.6088.6788.95Average scoreRoBERTa-large2481685.085.586.086.587.085.7586.1185.9686.0786.2086.1685.9986.21Average scoreRoBERTa-base2481688.088.589.089.588.5588.8488.8788.6388.9089.2489.2389.21Average scoreDeBERTaV3-base2481685.085.586.086.587.085.3085.8985.9986.4085.6786.2386.6086.45Average scoreRoBERTa-baseaLoRATriLoRAbcAdaLoRA2481685.085.586.086.587.086.1485.6486.1286.4186.1886.1386.7186.4486.1386.0586.3486.5686.2085.8386.0286.54Average scoreRoBERTa-baseMethodMethod + MPC oMethod + MPC dMethod + MPC n2481688.088.589.089.588.5288.6888.7188.9688.9888.9489.1589.0088.5188.8989.1489.0688.8088.5388.7788.71Average scoreRoBERTa-large2481688.589.089.590.088.8989.2089.2089.0389.2189.7089.4489.61Average scoreDeBERTaV3-base248168788899087.9588.6588.1788.9188.4588.7989.6489.4388.8188.8589.1089.6288.9989.0689.2189.47Average scoreDeBERTaV3-baseJOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021

10

DoRA [20] begins by decomposing the model weights W
into two components: magnitude and direction. The process
of adjusting these components is defined as follows:

SVDiff [73] modifies the singular values Σ of the original
weight matrix W by incorporating a diagonal “spectral shift”
matrix D. This adjustment is formulated as follows:

ϕ(W) = m

W + AB
∥W + AB∥c

,

(32)

where m ∈ R1×m represents the magnitude, and ∥ · ∥c is
the vector-wise norm of a matrix applied across each column.
Given that m is learnable during the training process, this
formula can be simplified as:

ϕ(W) = WD + ABD,

(33)

where D ∈ Rm×m is a diagonal matrix. We focus on the ex-
tension ∆W = ABD while disregarding the transformation
of the column space of W. Following the analysis similar to
that for LoRA derivatives, we can derive

I = A†∆W∗D†B† = A†UΣVTD†B†.

(34)

With the constraints A† = D1UT and D†B† = VD2, where
D1 ∈ Rr×n and D2 ∈ Rm×r are rectangle diagonal, we have

I = D1ΣD2, A = UD†

1, B = D†

2VTD†.

(35)

It is important to note
2 D†

BTB = D†TVD†T

2 D†

2VTD†, BBT = D†

2VTD†D†TVD†T
2 .
(36)
2) = r and the matrix product D†T
Since rank(D†
2 ∈ Rm×m
cannot be an identity matrix, both BTB and BBT are arbi-
trary matrices. Therefore, DoRA can at most impose semi-
orthogonal constraints on matrix A, while matrix B remains
unconstrained. Additionally, DoRA reconstructs W by scaling
its column space. Following an analysis similar to that used
for LoRA derivatives, it can be concluded that DoRA may
impose a semi-orthogonal constraint on one of the matrices,
either A or B, while leaving the other matrix without any
constraints. Furthermore, DoRA reconstructs W by scaling
its column space. Based on Proposition 2, it is concluded that
DoRA can lead to superior performance compared to LoRA,
as shown in Fig. 4 and also noted in [20], [42].

The Spectral Adapter [64] is another innovative example
within combination-based methods. Specifically, the Spectral
Adapter starts by decomposing the weight matrix W into
its SVD form W = UΣVT, and then introduces trainable
matrices A ∈ Rn×r and B ∈ Rm×r that are added to the top
r columns of the singular vectors U and V, with the form as

ϕ(W) = (cid:2)Ur + A Un−r

(cid:3) Σ (cid:2)Vr + B Vn−r

(cid:3)T

,

(37)

where Ur and Vr represent the top r columns of U and V,
and Un−r and Vn−r account for the remaining columns. The
Eq. (37) can be rewritten as

ϕ(W) = W + AΣVT + UΣBT + AΣB.

(38)

Therefore, Spectral Adapter can also be viewed as introducing
an additional term ∆W = AΣVT + UΣBT + AΣB. This
addition is not just a simple reconstruction of the original
subspace defined by the singular vectors but also acts as an
extension of that subspace.

ϕ(W) = Uh(Σ + D)VT,

(39)

where U and V represent the left and right singular vectors
of W, respectively, and h(·) denotes the nonlinear function
ReLU. This equation can be expanded to

ϕ(W) = UΣVT + U HΣ(D)VT

= W + ∆W.

(40)

In this context, HA(·) is an element-wise operator where
HΣ(D) = [max(Dij, −Σij)]n×m. Consequently, SVDiff not
only reconstructs but also extends the subspace. Moreover, we
can conclude that the approach, which selectively reconstructs
the singular values or vectors by introducing additional train-
able components rather than directly altering the singular com-
ponents, can be categorized as a combination-based method.

VI. CONCLUSION AND DISCUSSION

The adaptation of pre-trained foundation models for a
diverse array of downstream tasks has become a ubiquitous
practice in artificial intelligence. Given the extensive range
of tasks and the prohibitive costs associated, it is imprac-
tical to adjust all parameters comprehensively. In response,
the development of parameter-efficient fine-tuning techniques
(PEFT) has emerged, facilitating updates to the pre-trained
model weights in a manner that is significantly more resource-
efficient. Although methods of PEFT continue to proliferate,
a comprehensive understanding of their underlying math-
ematical principles and the variance in their performance
remains elusive. Therefore, in this work, we take the first
step by conceptualizing all PEFT methods from a decompo-
sition perspective, unifying them under the subspace tuning
methodology. The mathematical foundations underlying each
PEFT method are dissected, identifying that each represents a
distinct manipulation of the subspace. Inspired by theoretical
insights, we propose two novel PEFT methods. Extensive
experiments show that by training less than one thousandth of
the parameters, can approximate the effects of full fine-tuning.
Furthermore, we elucidate the reasons behind the per-
formance disparities among different methods. Our analy-
sis yields significant conclusions. The comparative analysis
of various PEFT strategies such as LoRA, AdaLoRA, and
FLoRA, reveals distinct patterns in their efficacy during
training. The more stringent the matrix pattern learning, the
more the model performance is constrained. We tested the
performance of nearly ten algorithms on three different large
pretrained models under four levels of parameter budgets, vali-
dating our conclusions with more than 3800 experimental runs.
Based on this analysis, we propose a framework that enhances
the learning of matrix patterns during model training. The
effectiveness of this framework has been confirmed through
more than 2000 experimental runs across three methods, four
parameter budgets, and three large pretrained models.

The significance of our findings extends beyond the imme-
diate scope of parameter-efficient fine-tuning. The principles

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021

11

underlying PEFT methods can be extrapolated to other do-
mains of artificial intelligence, such as transfer learning [22],
[44], multi-task learning [25], [74], and fast training [25], [75],
and also areas where computational resources are a limiting
factor, such as real-time systems [46] and embedded devices
[47]. By analyzing the theoretical aspects of PEFT methods
in different scenarios, we can comprehend the underlying
logic and, based on these theoretical insights, refine these
methods to further enhance their impact across related fields.
Additionally, the theoretical underpinnings of subspace tuning
present intriguing possibilities for further exploration in this
domain as well as others, potentially catalyzing advancements
and influencing developments across the broader artificial
intelligence landscape.

REFERENCES

[1] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., “Language mod-
els are few-shot learners,” Advances in neural information processing
systems, vol. 33, pp. 1877–1901, 2020.

[2] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al.,
“Language models are unsupervised multitask learners,” OpenAI blog,
vol. 1, no. 8, p. 9, 2019.

[3] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable
language supervision,” in International
visual models from natural
conference on machine learning. PMLR, 2021, pp. 8748–8763.
[4] J. D. M.-W. C. Kenton and L. K. Toutanova, “Bert: Pre-training of deep
bidirectional transformers for language understanding,” in Proceedings
of naacL-HLT, vol. 1. Minneapolis, Minnesota, 2019, p. 2.

[5] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis,
L. Zettlemoyer, and V. Stoyanov, “Roberta: A robustly optimized bert
pretraining approach,” arXiv preprint arXiv:1907.11692, 2019.

[6] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson,
T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,”
in Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2023, pp. 4015–4026.

[7] C. Si, X. Wang, X. Yang, and W. Shen, “Tendency-driven mutual
exclusivity for weakly supervised incremental semantic segmentation,”
in European Conference on Computer Vision. Springer, 2025, pp. 37–
54.

[8] K. Zhang and D. Liu, “Customized segment anything model for medical

image segmentation,” arXiv preprint arXiv:2304.13785, 2023.

[9] C. Zhang, L. Liu, Y. Cui, G. Huang, W. Lin, Y. Yang, and Y. Hu,
“A comprehensive survey on segment anything model for vision and
beyond,” arXiv preprint arXiv:2305.08196, 2023.
[10] L. B. Y. Ai, C. Ai, and R. Ai, “Gpt-4 technical report.”
[11] E. Waisberg, J. Ong, M. Masalkhi, S. A. Kamran, N. Zaman, P. Sarker,
A. G. Lee, and A. Tavakkoli, “Gpt-4: a new era of artificial intelligence
in medicine,” Irish Journal of Medical Science (1971-), vol. 192, no. 6,
pp. 3197–3200, 2023.

[12] R. Mao, G. Chen, X. Zhang, F. Guerin, and E. Cambria, “Gpteval: A
survey on assessments of chatgpt and gpt-4,” in Proceedings of the 2024
Joint International Conference on Computational Linguistics, Language
Resources and Evaluation (LREC-COLING 2024), 2024, pp. 7844–7866.
[13] J. Ma, Y. He, F. Li, L. Han, C. You, and B. Wang, “Segment anything in
medical images,” Nature Communications, vol. 15, no. 1, p. 654, 2024.
[14] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena,
Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning
with a unified text-to-text transformer,” Journal of machine learning
research, vol. 21, no. 140, pp. 1–67, 2020.

[15] X. Qiu, T. Sun, Y. Xu, Y. Shao, N. Dai, and X. Huang, “Pre-trained
language processing: A survey,” Science China

models for natural
Technological Sciences, vol. 63, no. 10, pp. 1872–1897, 2020.

[16] W. Chen, Z. Miao, and Q. Qiu, “Parameter-efficient tuning of large
convolutional models,” arXiv preprint arXiv:2403.00269, 2024.
[17] D. Guo, A. Rush, and Y. Kim, “Parameter-efficient transfer learning with
diff pruning,” in Annual Meeting of the Association for Computational
Linguistics, 2021.

[18] J. He, C. Zhou, X. Ma, T. Berg-Kirkpatrick, and G. Neubig, “Towards
a unified view of parameter-efficient transfer learning,” in International
Conference on Learning Representations, 2022. [Online]. Available:
https://openreview.net/forum?id=0RDcd5Axok

[19] E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang,
and W. Chen, “LoRA: Low-rank adaptation of large language models,”
in International Conference on Learning Representations, 2022.
[Online]. Available: https://openreview.net/forum?id=nZeVKeeFYf9
[20] S.-y. Liu, C.-Y. Wang, H. Yin, P. Molchanov, Y.-C. F. Wang, K.-T.
Cheng, and M.-H. Chen, “Dora: Weight-decomposed low-rank adap-
tation,” in Forty-first International Conference on Machine Learning,
2024.

[21] N. Ding, Y. Qin, G. Yang, F. Wei, Z. Yang, Y. Su, S. Hu, Y. Chen,
C.-M. Chan, W. Chen et al., “Parameter-efficient fine-tuning of large-
scale pre-trained language models,” Nature Machine Intelligence, vol. 5,
no. 3, pp. 220–235, 2023.

[22] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe,
A. Gesmundo, M. Attariyan, and S. Gelly, “Parameter-efficient transfer
learning for nlp,” in International conference on machine learning.
PMLR, 2019, pp. 2790–2799.

[23] S. Chen, C. Ge, Z. Tong, J. Wang, Y. Song, J. Wang, and P. Luo,
“Adaptformer: Adapting vision transformers for scalable visual recogni-
tion,” Advances in Neural Information Processing Systems, vol. 35, pp.
16 664–16 678, 2022.

[24] G. Luo, M. Huang, Y. Zhou, X. Sun, G. Jiang, Z. Wang, and R. Ji,
“Towards efficient visual adaption via structural re-parameterization,”
arXiv preprint arXiv:2302.08106, 2023.

[25] R. K. Mahabadi, S. Ruder, M. Dehghani, and J. Henderson, “Parameter-
efficient multi-task fine-tuning for transformers via shared hypernet-
works,” in Proceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International Joint Conference
on Natural Language Processing (Volume 1: Long Papers), 2021, pp.
565–576.

[26] R. Karimi Mahabadi, J. Henderson, and S. Ruder, “Compacter: Efficient
low-rank hypercomplex adapter layers,” Advances in Neural Information
Processing Systems, vol. 34, pp. 1022–1035, 2021.

[27] B. Lester, R. Al-Rfou, and N. Constant, “The power of scale for
parameter-efficient prompt tuning,” in Proceedings of the 2021 Con-
ference on Empirical Methods in Natural Language Processing, 2021,
pp. 3045–3059.

[28] A. Razdaibiedina, Y. Mao, M. Khabsa, M. Lewis, R. Hou, J. Ba,
and A. Almahairi, “Residual prompt tuning: improving prompt tuning
with residual reparameterization,” in The 61st Annual Meeting Of The
Association For Computational Linguistics, 2023.

[29] Y. Wang, J. Wu, T. Dabral, J. Zhang, G. Brown, C.-T. Lu, F. Liu,
Y. Liang, B. Pang, M. Bendersky et al., “Non-intrusive adaptation:
Input-centric parameter-efficient fine-tuning for versatile multimodal
modeling,” arXiv preprint arXiv:2310.12100, 2023.
[30] Z. Shi and A. Lipani, “DePT: Decomposed prompt

tuning for
parameter-efficient fine-tuning,” in The Twelfth International Conference
on Learning Representations, 2024.
[Online]. Available: https:
//openreview.net/forum?id=KjegfPGRde

[31] M. Fischer, A. Bartler, and B. Yang, “Prompt tuning for parameter-
efficient medical image segmentation,” Medical Image Analysis, vol. 91,
p. 103024, 2024.

[32] N. Hyeon-Woo, M. Ye-Bin, and T.-H. Oh, “Fedpara: Low-rank
hadamard product for communication-efficient federated learning,” in
International Conference on Learning Representations, 2022. [Online].
Available: https://openreview.net/forum?id=d71n4ftoCBy

[33] Z. Qiu, W. Liu, H. Feng, Y. Xue, Y. Feng, Z. Liu, D. Zhang, A. Weller,
and B. Sch¨olkopf, “Controlling text-to-image diffusion by orthogo-
nal finetuning,” Advances in Neural Information Processing Systems,
vol. 36, pp. 79 320–79 362, 2023.

[34] A. Renduchintala, T. Konuk, and O. Kuchaiev, “Tied-lora: Enhancing
parameter efficiency of lora with weight tying,” in Proceedings of the
2024 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies (Volume 1:
Long Papers), 2024, pp. 8686–8697.

[35] D. J. Kopiczko, T. Blankevoort, and Y. M. Asano, “VeRA: Vector-
based random matrix adaptation,” in The Twelfth International
Conference on Learning Representations, 2024. [Online]. Available:
https://openreview.net/forum?id=NjNfLdxr3A

[36] S.-Y. YEH, Y.-G. Hsieh, Z. Gao, B. B. Yang, G. Oh, and Y. Gong,
“Navigating text-to-image customization: From lycoris fine-tuning to
model evaluation,” in The Twelfth International Conference on Learning
Representations, 2023.

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021

12

[37] Q. Zhang, M. Chen, A. Bukharin, P. He, Y. Cheng, W. Chen, and
T. Zhao, “Adaptive budget allocation for parameter-efficient fine-tuning,”
in The Eleventh International Conference on Learning Representations,
2022.

[38] C. Si, X. Wang, X. Yang, Z. Xu, Q. Li, J. Dai, Y. Qiao, X. Yang, and
W. Shen, “Flora: Low-rank core space for n-dimension,” arXiv preprint
arXiv:2405.14739, 2024.

[39] C. Si, Z. Shi, S. Zhang, X. Yang, H. Pfister, and W. Shen, “Unleashing
the power of task-specific directions in parameter efficient fine-tuning,”
arXiv preprint arXiv:2409.01035, 2024.

[40] E. B. Zaken, Y. Goldberg, and S. Ravfogel, “Bitfit: Simple parameter-
efficient fine-tuning for transformer-based masked language-models,”
in Proceedings of
the Association for
the 60th Annual Meeting of
Computational Linguistics (Volume 2: Short Papers), 2022, pp. 1–9.

[41] N. G. Lawton, A. Kumar, G. Thattai, A. Galstyan, and G. Ver Steeg,
“Neural architecture search for parameter-efficient fine-tuning of large
pre-trained language models,” in The 61st Annual Meeting Of The
Association For Computational Linguistics, 2023.

[42] Z. Han, C. Gao, J. Liu, J. Zhang, and S. Q. Zhang, “Parameter-
large models: A comprehensive survey,”
efficient fine-tuning for
Transactions on Machine Learning Research, 2024. [Online]. Available:
https://openreview.net/forum?id=lIsCS8b6zj

[43] Z. Fu, H. Yang, A. M.-C. So, W. Lam, L. Bing, and N. Collier, “On the
effectiveness of parameter-efficient fine-tuning,” in Proceedings of the
AAAI Conference on Artificial Intelligence, vol. 37, no. 11, 2023, pp.
12 799–12 807.

[44] P. K. Mudrakarta, M. Sandler, A. Zhmoginov, and A. Howard, “K for
the price of 1: Parameter-efficient multi-task and transfer learning,” in
International Conference on Learning Representations, 2018.

[45] C. Si, Z. Jiang, X. Wang, Y. Wang, X. Yang, and W. Shen, “Partial
label learning with a partner,” in Proceedings of the AAAI Conference
on Artificial Intelligence, vol. 38, no. 13, 2024, pp. 15 029–15 037.
[46] M. Jovanovic and P. Voss, “Trends and challenges of real-time learn-
review,” arXiv preprint

ing in large language models: A critical
arXiv:2404.18311, 2024.

[47] T. Feng and S. Narayanan, “Peft-ser: On the use of parameter efficient
transfer learning approaches for speech emotion recognition using pre-
trained speech models,” in 2023 11th International Conference on
Affective Computing and Intelligent Interaction (ACII).
IEEE, 2023,
pp. 1–8.

[48] C. Si, Y. Jia, R. Wang, M.-L. Zhang, Y. Feng, and Q. Chongxiao, “Multi-
label classification with high-rank and high-order label correlations,”
IEEE Transactions on Knowledge and Data Engineering, 2023.
[49] S. Wang, L. Yu, and J. Li, “LoRA-GA: Low-rank adaptation with
gradient approximation,” in The Thirty-eighth Annual Conference on
Neural Information Processing Systems, 2024. [Online]. Available:
https://openreview.net/forum?id=VaLAWrLHJv

[50] F. Meng, Z. Wang, and M. Zhang, “PiSSA: Principal singular values
and singular vectors adaptation of large language models,” in The
Thirty-eighth Annual Conference on Neural Information Processing
Systems, 2024. [Online]. Available: https://openreview.net/forum?id=
6ZBHIEtdP4

[51] Z. Wang and J. Liang, “Lora-pro: Are low-rank adapters properly

optimized?” arXiv preprint arXiv:2407.18242, 2024.

[52] Y. Hao, Y. Cao, and L. Mou, “Flora: Low-rank adapters are secretly gra-
dient compressors,” in Forty-first International Conference on Machine
Learning, 2024.

[53] P. He, J. Gao, and W. Chen, “Debertav3: Improving deberta using
electra-style pre-training with gradient-disentangled embedding shar-
ing,” in The Eleventh International Conference on Learning Represen-
tations, 2021.

[54] Z. Peng, Z. Xu, Z. Zeng, X. Yang, and W. Shen, “Sam-parser: Fine-
tuning sam efficiently by parameter space reconstruction,” in Proceed-
ings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 5,
2024, pp. 4515–4523.

[55] H. Liu, D. Tam, M. Muqeeth, J. Mohta, T. Huang, M. Bansal, and C. A.
Raffel, “Few-shot parameter-efficient fine-tuning is better and cheaper
than in-context learning,” Advances in Neural Information Processing
Systems, vol. 35, pp. 1950–1965, 2022.

[56] X. L. Li and P. Liang, “Prefix-tuning: Optimizing continuous prompts
for generation,” in Proceedings of the 59th Annual Meeting of the
Association for Computational Linguistics and the 11th International
Joint Conference on Natural Language Processing (Volume 1: Long
Papers), 2021, pp. 4582–4597.

[57] T. Gao, A. Fisch, and D. Chen, “Making pre-trained language models
better few-shot learners,” in Proceedings of the 59th Annual Meeting of
the Association for Computational Linguistics and the 11th International

Joint Conference on Natural Language Processing (Volume 1: Long
Papers), 2021, pp. 3816–3830.

[58] Z. Tan, X. Zhang, S. Wang, and Y. Liu, “Msp: Multi-stage prompting for
making pre-trained language models better translators,” in Proceedings
of
the Association for Computational
Linguistics (Volume 1: Long Papers), 2022, pp. 6131–6142.

the 60th Annual Meeting of

[59] Y.-L. Sung, V. Nair, and C. A. Raffel, “Training neural networks
with fixed sparse masks,” Advances in Neural Information Processing
Systems, vol. 34, pp. 24 193–24 205, 2021.

[60] S. S. S. Das, R. H. Zhang, P. Shi, W. Yin, and R. Zhang, “Unified
low-resource sequence labeling by sample-aware dynamic sparse fine-
tuning,” in 2023 Conference on Empirical Methods in Natural Language
Processing, EMNLP 2023. Association for Computational Linguistics
(ACL), 2023, pp. 6998–7010.

[61] M. Gheini, X. Ren, and J. May, “Cross-attention is all you need: Adapt-
ing pretrained transformers for machine translation,” in Proceedings
of the 2021 Conference on Empirical Methods in Natural Language
Processing, 2021, pp. 1754–1765.

[62] H. He, J. Cai, J. Zhang, D. Tao, and B. Zhuang, “Sensitivity-aware
visual parameter-efficient fine-tuning,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2023, pp. 11 825–11 835.
[63] B. Liao, Y. Meng, and C. Monz, “Parameter-efficient fine-tuning without
introducing new latency,” in Proceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers),
2023, pp. 4242–4260.

[64] F. Zhang and M. Pilanci, “Spectral adapter: Fine-tuning in spectral
space,” in The Thirty-eighth Annual Conference on Neural Information
Processing Systems, 2024. [Online]. Available: https://openreview.net/
forum?id=UoxuaOGV6B

[65] A. Aghajanyan, S. Gupta, and L. Zettlemoyer, “Intrinsic dimensionality
explains the effectiveness of language model fine-tuning,” in Proceed-
ings of the 59th Annual Meeting of the Association for Computational
Linguistics and the 11th International Joint Conference on Natural
Language Processing (Volume 1: Long Papers), 2021, pp. 7319–7328.
[66] C. Li, H. Farkhoor, R. Liu, and J. Yosinski, “Measuring the intrinsic
dimension of objective landscapes,” in International Conference on
Learning Representations, 2018.

[67] R. Piziak and P. L. Odell, “Full rank factorization of matrices,” Mathe-

matics magazine, vol. 72, no. 3, pp. 193–201, 1999.

[68] J. G. Francis, “The qr transformation a unitary analogue to the lr
transformation—part 1,” The Computer Journal, vol. 4, no. 3, pp. 265–
271, 1961.

[69] V. N. Kublanovskaya, “On some algorithms for the solution of the
complete eigenvalue problem,” USSR Computational Mathematics and
Mathematical Physics, vol. 1, no. 3, pp. 637–657, 1962.

[70] C. Feng, M. He, Q. Tian, H. Yin, X. Zhao, H. Tang, and X. Wei,
“Trilora: Integrating svd for advanced style personalization in text-to-
image generation,” arXiv preprint arXiv:2405.11236, 2024.

[71] X. Hu, L. Chu, J. Pei, W. Liu, and J. Bian, “Model complexity of deep
learning: A survey,” Knowledge and Information Systems, vol. 63, pp.
2585–2619, 2021.

[72] J. Pfeiffer, A. Kamath, A. R¨uckl´e, K. Cho, and I. Gurevych, “Adapter-
fusion: Non-destructive task composition for transfer learning,” in
Proceedings of the 16th Conference of the European Chapter of the
Association for Computational Linguistics: Main Volume, 2021, pp. 487–
503.

[73] L. Han, Y. Li, H. Zhang, P. Milanfar, D. Metaxas, and F. Yang, “Svdiff:
Compact parameter space for diffusion fine-tuning,” in Proceedings of
the IEEE/CVF International Conference on Computer Vision, 2023, pp.
7323–7334.

[74] P. Liu, W. Yuan, J. Fu, Z. Jiang, H. Hayashi, and G. Neubig, “Pre-
train, prompt, and predict: A systematic survey of prompting methods
in natural language processing,” ACM Computing Surveys, vol. 55, no. 9,
pp. 1–35, 2023.

[75] A. R¨uckl´e, G. Geigle, M. Glockner, T. Beck, J. Pfeiffer, N. Reimers,
and I. Gurevych, “Adapterdrop: On the efficiency of adapters in trans-
formers,” in Proceedings of the 2021 Conference on Empirical Methods
in Natural Language Processing, 2021, pp. 7930–7946.

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021

13

TABLE II
RESULTS WITH ROBERTA-BASE [5] FINE-TUNED ON GLUE DEVELOPMENT SET.

Row Method

‰Params

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19

20
21
22
23
24
25
26
27
28
29
30
31
32

33
34
35
36
37
38
39
40
41
42
43
44
45

46
47
48
49
50
51
52
53
54
55
56
57
58

Fully FT
SAM-PARSER
(IA)3
SSL
SSB
BitFit
HAdapter
PAdapter
PA
LoRA
LoRA + MPCo
LoRA + MPCd
LoRA + MPCn
TriLoRA
TriLoRA + MPCo
AdaLoRA - MPCo
AdaLoRA
FLoRA
DoRA

HAdapter
PAdapter
PA
LoRA
LoRA + MPCo
LoRA + MPCd
LoRA + MPCn
TriLoRA
TriLoRA + MPCo
AdaLoRA - MPCo
AdaLoRA
FLoRA
DoRA

HAdapter
PAdapter
PA
LoRA
LoRA + MPCo
LoRA + MPCd
LoRA + MPCn
TriLoRA
TriLoRA + MPCo
AdaLoRA - MPCo
AdaLoRA
FLoRA
DoRA

HAdapter
PAdapter
PA
LoRA
LoRA + MPCo
LoRA + MPCd
LoRA + MPCn
TriLoRA
TriLoRA + MPCo
AdaLoRA - MPCo
AdaLoRA
FLoRA
DoRA

1000‰
0.44‰
0.44‰
0.22‰
0.66‰
0.82‰
2.50‰
2.43‰
2.65‰
2.65‰
2.65‰
2.65‰
2.65‰
2.65‰
2.65‰
2.65‰
2.65‰
2.65‰
3.32‰

4.87‰
4.80‰
5.31‰
5.31‰
5.31‰
5.31‰
5.31‰
5.31‰
5.31‰
5.31‰
5.31‰
5.31‰
5.97‰

9.59‰
9.52‰
10.62‰
10.62‰
10.62‰
10.62‰
10.62‰
10.62‰
10.62‰
10.62‰
10.62‰
10.64‰
11.28‰

19.03‰
18.96‰
21.24‰
21.24‰
21.24‰
21.24‰
21.24‰
21.24‰
21.24‰
21.24‰
21.24‰
21.38‰
21.90‰

MNLI
Acc
87.62
54.92
84.83
83.45
85.80
85.29
87.45
87.11
87.55
87.20
86.96
87.03
87.55
86.81
87.48
87.51
87.31
87.31
86.74

87.34
87.22
87.32
87.13
87.02
87.15
87.32
87.45
87.90
87.39
87.43
87.59
87.06

87.74
86.90
87.50
87.38
87.50
87.31
87.50
87.44
87.97
87.31
87.13
87.43
87.14

87.31
86.64
87.55
87.49
87.44
87.34
87.55
87.54
88.21
87.32
87.33
87.64
86.98

SST-2
Acc
94.84
85.09
94.15
93.81
94.61
94.29
94.72
94.15
94.38
94.38
94.72
94.50
94.38
94.61
94.27
94.15
94.72
94.38
94.50

94.95
94.84
94.27
94.38
94.38
94.27
94.27
93.81
94.84
94.50
94.50
93.92
94.38

94.04
94.50
94.61
94.84
94.61
94.84
94.61
94.84
94.72
94.61
94.61
94.27
94.50

94.72
94.84
94.50
94.27
94.95
94.50
94.50
94.50
94.61
94.61
94.50
94.95
94.61

CoLA QQP QNLI
Acc
Acc
Mcc
92.80
91.87
63.58
70.79
75.92
39.85
90.39
87.92
60.14
89.20
87.30
56.02
91.20
88.65
60.92
90.39
88.10
59.58
92.71
90.29
63.88
92.71
89.95
62.74
92.81
90.03
64.23
92.07
89.25
65.61
92.00
89.24
64.08
92.13
89.19
64.24
92.81
90.03
64.23
91.82
89.61
64.47
92.55
89.93
63.97
92.97
90.14
62.35
92.81
89.77
64.33
92.77
89.97
64.09
91.95
90.28
66.19

RTE MRPC
Acc
78.80
59.57
76.17
74.01
78.53
78.84
80.14
80.14
80.51
81.59
81.23
81.51
80.51
76.53
78.34
80.51
81.95
82.67
79.78

Acc
90.20
74.26
87.75
86.76
87.75
88.73
89.22
87.99
89.22
87.99
90.20
89.46
89.22
88.24
88.24
87.75
88.24
87.75
88.48

63.68
64.91
62.43
62.52
64.58
63.68
62.43
63.85
62.20
61.84
61.74
62.13
66.19

61.53
66.52
62.89
64.58
65.38
64.43
62.89
62.91
65.06
62.45
62.61
63.31
66.06

63.91
64.55
65.79
64.70
62.62
65.33
65.79
64.72
62.89
62.41
63.91
65.17
65.30

90.55
89.95
90.26
89.40
89.47
89.44
90.26
90.30
90.24
90.12
89.76
90.28
90.67

89.27
90.05
89.51
89.39
89.55
89.41
89.51
90.70
90.38
89.41
89.49
90.38
91.02

90.80
90.22
90.93
89.39
89.48
89.26
90.93
90.98
90.53
90.19
89.30
90.90
90.92

92.81
92.38
92.68
92.20
92.33
92.44
92.68
92.42
92.60
92.64
92.90
92.73
92.15

92.57
92.46
92.73
92.47
93.19
92.57
92.73
92.82
93.01
92.88
92.62
92.75
92.13

92.49
92.73
92.68
92.33
93.15
92.82
92.68
92.93
93.01
92.90
92.62
92.53
92.90

81.23
79.06
79.42
79.42
81.23
81.59
79.42
80.14
83.03
82.31
82.67
82.67
80.87

80.14
80.87
80.87
80.14
83.03
82.31
80.87
80.14
81.95
81.74
81.95
81.59
81.95

79.78
76.17
80.87
81.59
82.67
83.03
80.87
80.87
82.67
81.59
82.31
79.78
78.70

89.71
88.75
89.22
88.97
88.97
88.73
89.22
88.97
88.73
89.71
89.71
88.73
88.73

89.46
88.24
89.46
89.21
89.22
88.73
89.46
88.73
89.22
88.73
88.97
90.44
88.73

89.95
90.44
89.22
90.44
90.20
88.97
89.22
88.73
88.97
88.97
88.97
89.71
88.48

STS-B
Corr
91.23
60.65
90.23
89.52
90.31
90.32
90.80
90.13
90.85
91.01
91.03
91.00
90.85
90.31
90.55
90.62
90.48
90.77
91.01

91.21
90.31
91.07
91.11
91.02
91.08
91.07
90.17
90.26
90.34
90.55
90.78
90.94

90.84
90.13
90.59
90.96
91.21
91.09
90.59
90.37
90.51
90.51
90.53
90.82
90.96

91.00
90.66
90.74
91.03
90.99
91.22
90.74
90.93
90.67
90.57
90.75
90.75
91.00

All
Avg.
86.37
65.13
83.91
82.51
84.72
84.44
86.15
85.62
86.20
86.14
86.18
86.13
86.20
85.30
85.67
85.75
86.20
86.21
86.12

86.44
85.93
85.83
85.64
86.13
86.05
85.83
85.89
86.23
86.11
86.16
86.10
86.37

85.70
86.21
86.02
86.12
86.71
86.34
86.02
85.99
86.60
85.96
85.99
86.37
86.56

86.25
85.78
86.54
86.41
86.44
86.56
86.54
86.40
86.45
86.07
86.21
86.37
86.11

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021

14

TABLE III
RESULTS WITH DEBERTAV3-BASE [53] FINE-TUNED ON GLUE DEVELOPMENT SET.

Row Method

‰Params

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19

20
21
22
23
24
25
26
27
28
29
30
31
32

33
34
35
36
37
38
39
40
41
42
43
44
45

46
47
48
49
50
51
52
53
54
55
56
57
58

Fully FT
SAM-PARSER
(IA)3
SSL
SSB
BitFit
HAdapter
PAdapter
PA
LoRA
LoRA + MPCo
LoRA + MPCd
LoRA + MPCn
TriLoRA
TriLoRA + MPCo
AdaLoRA - MPCo
AdaLoRA
FLoRA
DoRA

HAdapter
PAdapter
PA
LoRA
LoRA + MPCo
LoRA + MPCd
LoRA + MPCn
TriLoRA
TriLoRA + MPCo
AdaLoRA - MPCo
AdaLoRA
FLoRA
DoRA

HAdapter
PAdapter
PA
LoRA
LoRA + MPCo
LoRA + MPCd
LoRA + MPCn
TriLoRA
TriLoRA + MPCo
AdaLoRA - MPCo
AdaLoRA
FLoRA
DoRA

HAdapter
PAdapter
PA
LoRA
LoRA + MPCo
LoRA + MPCd
LoRA + MPCn
TriLoRA
TriLoRA + MPCo
AdaLoRA - MPCo
AdaLoRA
FLoRA
DoRA

1000‰
0.30‰
0.30‰
0.15‰
0.45‰
0.54‰
1.68‰
1.63‰
1.79‰
1.79‰
1.79‰
1.79‰
1.79‰
1.79‰
1.79‰
1.79‰
1.79‰
1.79‰
2.23‰

3.32‰
3.26‰
3.61‰
3.61‰
3.61‰
3.61‰
3.61‰
3.61‰
3.61‰
3.61‰
3.61‰
3.61‰
4.06‰

6.63‰
6.41‰
7.23‰
7.23‰
7.23‰
7.23‰
7.23‰
7.23‰
7.23‰
7.23‰
7.23‰
7.24‰
7.66‰

12.93‰
12.88‰
14.43‰
14.43‰
14.43‰
14.43‰
14.43‰
14.43‰
14.43‰
14.43‰
14.43‰
14.52‰
14.87‰

MNLI
Acc
89.90
68.32
89.44
88.35
89.86
89.37
90.10
89.89
90.24
90.03
89.99
90.06
90.24
90.32
90.39
90.34
90.40
90.60
90.21

90.12
90.15
90.23
89.90
89.92
89.93
90.23
90.37
90.55
90.40
90.27
90.45
90.12

90.13
90.33
90.18
89.80
90.60
90.20
90.18
90.28
90.60
90.38
90.38
90.82
89.67

89.81
89.82
90.35
89.49
90.32
90.33
90.35
90.20
90.59
90.23
90.21
90.09
90.17

SST-2
Acc
95.63
84.28
95.52
95.07
95.53
94.84
95.41
94.72
95.64
93.92
93.61
95.30
95.64
95.87
95.87
95.10
95.80
96.00
94.38

95.30
95.53
96.11
94.15
95.06
95.18
96.11
96.10
96.10
95.76
95.76
95.76
96.22

95.53
95.61
95.99
93.69
95.64
95.53
95.99
95.99
96.10
95.99
95.87
96.21
94.61

95.41
94.84
95.64
93.78
95.98
95.64
95.64
95.87
96.22
96.10
95.87
95.87
93.92

CoLA QQP QNLI
Acc
Acc
Mcc
94.03
92.40
69.19
75.86
81.91
55.21
91.80
89.01
67.01
90.10
88.19
66.64
93.41
89.87
67.82
92.24
88.41
66.96
93.52
91.19
67.65
93.87
91.05
69.06
94.25
91.40
71.24
93.37
90.61
69.15
94.98
91.25
69.93
94.34
91.29
69.91
94.25
91.40
71.24
94.25
91.38
69.61
94.25
91.38
70.00
94.10
90.98
69.51
94.23
91.43
69.98
94.46
91.40
70.20
93.26
90.84
69.33

RTE MRPC
Acc
83.75
66.06
79.42
82.31
83.75
78.80
83.39
84.48
87.36
85.56
86.64
88.08
87.36
87.00
88.09
87.01
87.36
88.81
86.94

Acc
89.46
76.47
88.23
88.68
88.72
87.75
89.25
89.71
90.20
90.19
89.70
89.95
90.20
91.17
92.16
89.95
90.43
90.93
90.19

67.87
69.48
69.78
68.87
69.55
69.82
69.78
69.94
72.37
69.86
69.58
70.72
68.24

68.64
68.77
70.69
69.30
72.19
70.06
70.69
68.28
70.78
69.12
71.45
72.05
69.08

69.96
70.69
71.45
71.61
72.26
72.99
71.45
68.64
72.71
68.49
72.83
70.71
69.92

91.30
91.27
91.77
91.16
92.10
92.13
91.77
91.88
91.93
91.40
91.47
91.44
91.55

91.27
91.40
91.80
91.78
92.29
92.23
91.80
91.97
91.96
91.36
91.19
91.94
91.80

92.25
92.31
92.30
92.33
92.30
92.51
92.30
92.25
92.15
90.72
90.87
91.67
92.47

93.76
93.98
94.27
93.59
93.66
93.92
94.27
94.51
94.55
94.53
94.65
94.20
94.65

94.11
94.29
94.34
92.97
93.75
93.78
94.34
94.47
94.51
94.23
94.36
94.60
93.23

94.09
94.09
94.38
93.92
93.73
94.16
94.38
94.36
94.62
94.31
94.47
94.47
93.21

85.56
84.12
88.08
89.89
87.72
87.73
88.08
88.45
88.45
87.00
89.53
89.27
89.53

84.48
85.20
88.81
85.70
89.53
88.81
88.81
89.53
89.17
87.72
88.09
89.53
87.33

86.28
86.28
88.81
87.36
88.08
89.53
88.81
88.81
88.08
87.72
87.72
89.53
87.00

89.22
89.22
90.20
90.19
90.93
90.44
90.20
90.93
92.16
90.20
91.42
90.93
92.40

89.95
89.46
90.20
90.68
91.17
90.20
90.20
91.67
90.93
90.93
90.69
91.18
90.68

89.70
89.21
90.93
91.17
91.17
90.20
90.93
90.69
90.93
90.20
90.44
90.93
90.68

STS-B
Corr
91.60
82.82
90.79
90.13
90.94
91.35
91.31
91.38
91.57
90.75
91.53
91.53
91.57
91.48
91.53
91.40
91.63
91.96
91.34

91.30
91.52
92.03
91.46
91.34
91.65
92.03
91.42
91.52
91.58
91.23
91.35
91.29

91.48
91.54
91.68
91.62
91.91
91.99
91.68
91.43
91.49
91.20
91.84
92.04
91.73

91.93
91.54
91.89
91.63
91.61
91.61
91.89
91.39
91.56
91.30
91.23
91.58
91.73

All
Avg.
88.24
73.87
86.40
86.18
87.49
86.21
87.73
88.02
88.99
87.95
88.45
88.81
88.99
88.89
89.21
88.55
88.90
89.30
88.31

88.05
88.16
89.06
88.65
88.79
88.85
89.06
89.20
89.70
88.84
89.24
89.27
89.25

88.19
88.32
89.21
88.17
89.64
89.10
89.21
89.20
89.44
88.87
89.23
89.80
88.49

88.68
88.60
89.47
88.91
89.43
89.62
89.47
89.03
89.61
88.63
89.21
89.36
88.64

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021

15

TABLE IV
RESULTS WITH ROBERTA-LARGE [5] FINE-TUNED ON GLUE DEVELOPMENT SET.

Row Method

‰Params

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19

20
21
22
23
24
25
26
27
28
29
30
31
32

33
34
35
36
37
38
39
40
41
42
43
44
45

46
47
48
49
50
51
52
53
54
55
56
57
58

Fully FT
SAM-PARSER
(IA)3
SSL
SSB
BitFit
HAdapter
PAdapter
PA
LoRA
LoRA + MPCo
LoRA + MPCd
LoRA + MPCn
TriLoRA
TriLoRA + MPCo
AdaLoRA - MPCo
AdaLoRA
FLoRA
DoRA

HAdapter
PAdapter
PA
LoRA
LoRA + MPCo
LoRA + MPCd
LoRA + MPCn
TriLoRA
TriLoRA + MPCo
AdaLoRA - MPCo
AdaLoRA
FLoRA
DoRA

HAdapter
PAdapter
PA
LoRA
LoRA + MPCo
LoRA + MPCd
LoRA + MPCn
TriLoRA
TriLoRA + MPCo
AdaLoRA - MPCo
AdaLoRA
FLoRA
DoRA

HAdapter
PAdapter
PA
LoRA
LoRA + MPCo
LoRA + MPCd
LoRA + MPCn
TriLoRA
TriLoRA + MPCo
AdaLoRA - MPCo
AdaLoRA
FLoRA
DoRA

1000‰
0.42‰
0.42‰
0.21‰
0.62‰
0.76‰
2.35‰
2.29‰
2.49‰
2.49‰
2.49‰
2.49‰
2.49‰
2.49‰
2.49‰
2.49‰
2.49‰
2.49‰
3.12‰

4.57‰
4.50‰
4.98‰
4.98‰
4.98‰
4.98‰
4.98‰
4.98‰
4.98‰
4.98‰
4.98‰
4.99‰
5.61‰

9.00‰
8.93‰
9.97‰
9.97‰
9.97‰
9.97‰
9.97‰
9.97‰
9.97‰
9.97‰
9.97‰
9.98‰
10.59‰

17.87‰
17.80‰
19.94‰
19.94‰
19.94‰
19.94‰
19.94‰
19.94‰
19.94‰
19.94‰
19.94‰
19.98‰
20.56‰

MNLI
Acc
90.29
52.43
90.11
89.55
90.38
90.15
90.66
90.39
90.79
90.41
90.73
90.50
90.79
90.12
90.85
90.59
90.69
90.62
90.60

90.64
90.31
90.52
90.68
90.68
90.72
90.52
90.51
90.96
90.65
90.86
90.41
90.61

90.57
90.29
90.44
90.73
90.66
90.64
90.44
90.41
90.86
90.83
90.65
90.51
90.67

90.42
90.38
90.31
90.69
90.77
90.77
90.31
90.58
90.83
90.64
90.50
90.61
90.72

SST-2
Acc
96.41
87.50
95.99
95.87
95.64
96.22
96.22
96.22
96.33
95.99
96.33
96.33
96.33
95.87
96.10
96.10
96.22
96.10
96.22

96.44
96.22
95.87
96.10
95.99
96.21
95.87
96.10
95.76
96.22
95.76
95.99
96.79

96.10
95.76
96.67
95.87
95.99
96.22
96.67
96.67
96.10
96.22
96.10
95.87
95.99

96.10
94.84
96.33
95.87
96.10
96.33
96.33
95.99
96.44
96.33
95.87
96.10
96.22

CoLA QQP QNLI
Acc
Acc
Mcc
94.74
92.24
68.02
66.47
76.28
42.24
93.28
89.55
67.01
92.24
89.07
66.92
93.78
90.19
67.26
94.12
89.50
68.53
94.78
90.82
67.25
94.47
90.47
67.25
94.80
90.91
66.91
94.05
90.75
67.83
94.55
91.01
68.10
94.60
91.15
65.29
94.80
90.91
66.91
94.69
90.58
67.69
94.87
90.96
68.96
94.67
91.16
67.87
94.69
91.01
67.08
94.67
91.16
68.97
94.47
91.23
66.62

RTE MRPC
Acc
86.66
60.29
89.17
86.64
88.45
84.15
86.28
88.44
87.73
87.72
88.80
86.64
87.73
85.20
88.45
88.45
89.53
87.55
88.08

Acc
90.92
75.25
88.97
87.50
89.46
89.95
90.20
89.95
90.44
89.46
89.95
91.18
90.44
90.69
89.46
89.71
90.68
89.46
90.93

67.51
68.17
66.56
68.36
68.95
69.55
66.56
67.98
67.26
66.03
65.39
69.19
68.22

68.57
68.47
66.93
67.83
71.56
69.23
66.93
68.01
66.76
68.36
65.96
67.43
66.87

67.32
67.70
66.52
68.36
68.74
68.57
66.52
69.04
68.57
65.83
67.43
70.16
68.33

91.13
89.84
89.42
91.16
91.30
91.20
89.42
91.20
91.08
91.24
90.91
91.16
91.57

91.62
91.39
91.03
91.28
91.43
91.22
91.03
91.46
91.15
91.25
90.80
91.30
91.69

91.65
91.60
91.16
91.35
91.44
91.20
91.16
91.73
91.27
91.39
90.53
90.57
91.80

94.65
94.51
94.75
94.60
94.58
94.47
94.75
94.33
95.06
94.86
94.87
94.55
94.44

94.73
94.60
94.91
94.73
94.65
94.91
94.91
94.58
95.08
94.82
94.73
94.80
94.75

94.78
94.67
94.73
94.62
94.97
94.69
94.73
94.93
95.02
94.86
94.98
94.76
94.98

88.08
88.44
87.72
85.20
86.64
86.28
87.72
86.64
88.81
86.28
89.17
87.73
86.64

87.00
87.72
87.00
86.28
85.92
87.00
87.00
86.64
88.08
88.45
89.53
88.80
87.00

89.89
88.44
86.64
88.08
87.73
87.73
86.64
88.08
88.81
87.73
90.25
89.53
88.08

90.93
90.93
91.18
90.93
91.42
90.44
91.18
91.42
89.95
90.44
89.95
89.71
91.18

90.93
90.93
90.93
90.69
90.69
91.67
90.93
89.95
90.93
91.42
89.71
90.69
91.42

89.71
88.48
91.18
90.69
89.95
90.69
91.18
90.20
89.95
90.20
90.20
89.95
90.44

STS-B
Corr
92.44
54.27
88.60
86.19
89.82
91.68
92.30
92.38
92.46
91.92
92.36
92.38
92.46
91.43
91.45
92.08
92.16
92.26
92.00

92.21
92.21
92.24
92.43
92.35
92.27
92.24
91.86
91.80
92.05
91.87
92.04
92.55

92.49
92.61
92.23
92.30
92.31
92.24
92.23
92.35
91.91
91.92
91.90
92.18
92.30

92.50
92.30
92.79
92.05
92.27
92.52
92.79
92.00
92.05
91.84
91.83
92.24
92.65

All
Avg.
88.97
64.34
87.84
86.75
88.12
88.04
88.56
88.70
88.80
88.52
88.98
88.51
88.80
88.28
88.89
88.83
89.01
88.85
88.77

88.95
88.83
88.53
88.68
88.94
88.89
88.53
88.76
88.84
88.47
88.60
88.85
89.00

89.00
88.97
88.77
88.71
89.15
89.14
88.77
88.76
88.86
89.16
88.67
88.95
88.84

89.05
88.55
88.71
88.96
89.00
89.06
88.71
89.07
89.12
88.60
88.95
89.24
89.15

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021

16

Dataset

Task

# Train

# Dev

# Test

# Label

Metrics

TABLE V
DETAILS OF GLUE DATASET.

CoLA

SST-2

Acceptability

Single-Sentence Classification
2

1k

1k

8.5k

Sentiment

67k

872

1.8k

Similarity and Paraphrase

MRPC

Paraphrase

QQP

Paraphrase

STS-B

Similarity

3.7k

364k

7k

408

40k

1.5k

1.7k

391k

1.4k

Natural Language Inference

MNLI

NLI

QNLI

QA/NLI

RTE

NLI

393k

108k

2.5k

20k

5.7k

276

20k

5.7k

3k

2

2

2

1

3

2

2

Matthews corr

Accuracy

Accuracy / F1

Accuracy / F1

Pearson/ Spearman Corr

Accuracy

Accuracy

Accuracy

TABLE VI
HYPER-PARAMETER SETUP FOR GLUE BENCHMARK.

Dataset
learning rate
batch size
#epochs

MNLI
5 × 10−4
32
7

RTE
1.2 × 10−3
32
50

QNLI
1.2 × 10−3
32
5

MRPC
1 × 10−3
32
30

QQP
5 × 10−4
32
5

SST-2
8 × 10−4
32
24

CoLA
5 × 10−4
32
25

STS-B
2.2 × 10−3
32
25

