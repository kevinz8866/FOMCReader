ô»
È
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
û
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
®
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.12v2.10.0-76-gfdfc646704c8á
¬
*Adam/transformer_encoder_1/dense_90/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/transformer_encoder_1/dense_90/bias/v
¥
>Adam/transformer_encoder_1/dense_90/bias/v/Read/ReadVariableOpReadVariableOp*Adam/transformer_encoder_1/dense_90/bias/v*
_output_shapes
:*
dtype0
µ
,Adam/transformer_encoder_1/dense_90/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*=
shared_name.,Adam/transformer_encoder_1/dense_90/kernel/v
®
@Adam/transformer_encoder_1/dense_90/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/transformer_encoder_1/dense_90/kernel/v*
_output_shapes
:	*
dtype0

"Adam/layer_normalization_26/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/layer_normalization_26/beta/v

6Adam/layer_normalization_26/beta/v/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_26/beta/v*
_output_shapes
:@*
dtype0

#Adam/layer_normalization_26/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/layer_normalization_26/gamma/v

7Adam/layer_normalization_26/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_26/gamma/v*
_output_shapes
:@*
dtype0
z
Adam/Variable/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_nameAdam/Variable/v
s
#Adam/Variable/v/Read/ReadVariableOpReadVariableOpAdam/Variable/v*
_output_shapes

:@@*
dtype0
~
Adam/Variable/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*"
shared_nameAdam/Variable/v_1
w
%Adam/Variable/v_1/Read/ReadVariableOpReadVariableOpAdam/Variable/v_1*
_output_shapes

:@@*
dtype0
~
Adam/Variable/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*"
shared_nameAdam/Variable/v_2
w
%Adam/Variable/v_2/Read/ReadVariableOpReadVariableOpAdam/Variable/v_2*
_output_shapes

:@@*
dtype0

Adam/dense_89/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_89/bias/v
y
(Adam/dense_89/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_89/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_89/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_89/kernel/v

*Adam/dense_89/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_89/kernel/v*
_output_shapes

:@@*
dtype0
Å
4Adam/transformer_encoder_1/embedding_38/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'@*E
shared_name64Adam/transformer_encoder_1/embedding_38/embeddings/v
¾
HAdam/transformer_encoder_1/embedding_38/embeddings/v/Read/ReadVariableOpReadVariableOp4Adam/transformer_encoder_1/embedding_38/embeddings/v*
_output_shapes
:	'@*
dtype0
¬
*Adam/transformer_encoder_1/dense_90/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/transformer_encoder_1/dense_90/bias/m
¥
>Adam/transformer_encoder_1/dense_90/bias/m/Read/ReadVariableOpReadVariableOp*Adam/transformer_encoder_1/dense_90/bias/m*
_output_shapes
:*
dtype0
µ
,Adam/transformer_encoder_1/dense_90/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*=
shared_name.,Adam/transformer_encoder_1/dense_90/kernel/m
®
@Adam/transformer_encoder_1/dense_90/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/transformer_encoder_1/dense_90/kernel/m*
_output_shapes
:	*
dtype0

"Adam/layer_normalization_26/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/layer_normalization_26/beta/m

6Adam/layer_normalization_26/beta/m/Read/ReadVariableOpReadVariableOp"Adam/layer_normalization_26/beta/m*
_output_shapes
:@*
dtype0

#Adam/layer_normalization_26/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/layer_normalization_26/gamma/m

7Adam/layer_normalization_26/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/layer_normalization_26/gamma/m*
_output_shapes
:@*
dtype0
z
Adam/Variable/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_nameAdam/Variable/m
s
#Adam/Variable/m/Read/ReadVariableOpReadVariableOpAdam/Variable/m*
_output_shapes

:@@*
dtype0
~
Adam/Variable/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*"
shared_nameAdam/Variable/m_1
w
%Adam/Variable/m_1/Read/ReadVariableOpReadVariableOpAdam/Variable/m_1*
_output_shapes

:@@*
dtype0
~
Adam/Variable/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*"
shared_nameAdam/Variable/m_2
w
%Adam/Variable/m_2/Read/ReadVariableOpReadVariableOpAdam/Variable/m_2*
_output_shapes

:@@*
dtype0

Adam/dense_89/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_89/bias/m
y
(Adam/dense_89/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_89/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_89/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_89/kernel/m

*Adam/dense_89/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_89/kernel/m*
_output_shapes

:@@*
dtype0
Å
4Adam/transformer_encoder_1/embedding_38/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'@*E
shared_name64Adam/transformer_encoder_1/embedding_38/embeddings/m
¾
HAdam/transformer_encoder_1/embedding_38/embeddings/m/Read/ReadVariableOpReadVariableOp4Adam/transformer_encoder_1/embedding_38/embeddings/m*
_output_shapes
:	'@*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

#transformer_encoder_1/dense_90/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#transformer_encoder_1/dense_90/bias

7transformer_encoder_1/dense_90/bias/Read/ReadVariableOpReadVariableOp#transformer_encoder_1/dense_90/bias*
_output_shapes
:*
dtype0
§
%transformer_encoder_1/dense_90/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*6
shared_name'%transformer_encoder_1/dense_90/kernel
 
9transformer_encoder_1/dense_90/kernel/Read/ReadVariableOpReadVariableOp%transformer_encoder_1/dense_90/kernel*
_output_shapes
:	*
dtype0

layer_normalization_26/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_26/beta

/layer_normalization_26/beta/Read/ReadVariableOpReadVariableOplayer_normalization_26/beta*
_output_shapes
:@*
dtype0

layer_normalization_26/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namelayer_normalization_26/gamma

0layer_normalization_26/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_26/gamma*
_output_shapes
:@*
dtype0
l
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_name
Variable
e
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes

:@@*
dtype0
p

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_name
Variable_1
i
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

:@@*
dtype0
p

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_name
Variable_2
i
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes

:@@*
dtype0
r
dense_89/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_89/bias
k
!dense_89/bias/Read/ReadVariableOpReadVariableOpdense_89/bias*
_output_shapes
:@*
dtype0
z
dense_89/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_89/kernel
s
#dense_89/kernel/Read/ReadVariableOpReadVariableOpdense_89/kernel*
_output_shapes

:@@*
dtype0
·
-transformer_encoder_1/embedding_38/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'@*>
shared_name/-transformer_encoder_1/embedding_38/embeddings
°
Atransformer_encoder_1/embedding_38/embeddings/Read/ReadVariableOpReadVariableOp-transformer_encoder_1/embedding_38/embeddings*
_output_shapes
:	'@*
dtype0

serving_default_input_1Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
Á
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1-transformer_encoder_1/embedding_38/embeddings
Variable_2
Variable_1Variablelayer_normalization_26/gammalayer_normalization_26/betadense_89/kerneldense_89/bias%transformer_encoder_1/dense_90/kernel#transformer_encoder_1/dense_90/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_273820

NoOpNoOp
´J
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ïI
valueåIBâI BÛI
þ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
embedding_layer
	encoder


classifier
	optimizer

signatures*
J
0
1
2
3
4
5
6
7
8
9*
J
0
1
2
3
4
5
6
7
8
9*
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
 
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

embeddings*
È
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,ff_layer
-
self_atten
.
layer_norm
/call*
¦
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias*

6iter

7beta_1

8beta_2
	9decay
:learning_ratemmmmmmmmmmvvvvvvvvvv *

;serving_default* 
mg
VARIABLE_VALUE-transformer_encoder_1/embedding_38/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_89/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_89/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_2&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_1&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEVariable&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUElayer_normalization_26/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElayer_normalization_26/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%transformer_encoder_1/dense_90/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#transformer_encoder_1/dense_90/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
	1

2*

<0
=1*
* 
* 
* 
* 
* 
* 

0*

0*
* 

>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 
5
0
1
2
3
4
5
6*
5
0
1
2
3
4
5
6*
* 

Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

Jtrace_0* 

Ktrace_0* 
¦
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses

kernel
bias*
Ò
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses
k_weight
v_weight
q_weight
Xattn_mtx
Ycall*
¯
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses
`axis
	gamma
beta*

atrace_0* 

0
1*

0
1*
* 

bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

gtrace_0* 

htrace_0* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
8
i	variables
j	keras_api
	ktotal
	lcount*
H
m	variables
n	keras_api
	ototal
	pcount
q
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 

,0
-1
.2*
* 
* 
* 
* 
* 

0
1*

0
1*
* 

rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*
* 
* 

0
1
2*

0
1
2*
* 

wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*
* 
* 

|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

trace_0* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

k0
l1*

i	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

o0
p1*

m	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
	
X0* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

VARIABLE_VALUE4Adam/transformer_encoder_1/embedding_38/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_89/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_89/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEAdam/Variable/m_2Bvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEAdam/Variable/m_1Bvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/Variable/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/layer_normalization_26/gamma/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/layer_normalization_26/beta/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/transformer_encoder_1/dense_90/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/transformer_encoder_1/dense_90/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE4Adam/transformer_encoder_1/embedding_38/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_89/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_89/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEAdam/Variable/v_2Bvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEAdam/Variable/v_1Bvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/Variable/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/layer_normalization_26/gamma/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/layer_normalization_26/beta/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/transformer_encoder_1/dense_90/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/transformer_encoder_1/dense_90/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ª
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAtransformer_encoder_1/embedding_38/embeddings/Read/ReadVariableOp#dense_89/kernel/Read/ReadVariableOp!dense_89/bias/Read/ReadVariableOpVariable_2/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable/Read/ReadVariableOp0layer_normalization_26/gamma/Read/ReadVariableOp/layer_normalization_26/beta/Read/ReadVariableOp9transformer_encoder_1/dense_90/kernel/Read/ReadVariableOp7transformer_encoder_1/dense_90/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpHAdam/transformer_encoder_1/embedding_38/embeddings/m/Read/ReadVariableOp*Adam/dense_89/kernel/m/Read/ReadVariableOp(Adam/dense_89/bias/m/Read/ReadVariableOp%Adam/Variable/m_2/Read/ReadVariableOp%Adam/Variable/m_1/Read/ReadVariableOp#Adam/Variable/m/Read/ReadVariableOp7Adam/layer_normalization_26/gamma/m/Read/ReadVariableOp6Adam/layer_normalization_26/beta/m/Read/ReadVariableOp@Adam/transformer_encoder_1/dense_90/kernel/m/Read/ReadVariableOp>Adam/transformer_encoder_1/dense_90/bias/m/Read/ReadVariableOpHAdam/transformer_encoder_1/embedding_38/embeddings/v/Read/ReadVariableOp*Adam/dense_89/kernel/v/Read/ReadVariableOp(Adam/dense_89/bias/v/Read/ReadVariableOp%Adam/Variable/v_2/Read/ReadVariableOp%Adam/Variable/v_1/Read/ReadVariableOp#Adam/Variable/v/Read/ReadVariableOp7Adam/layer_normalization_26/gamma/v/Read/ReadVariableOp6Adam/layer_normalization_26/beta/v/Read/ReadVariableOp@Adam/transformer_encoder_1/dense_90/kernel/v/Read/ReadVariableOp>Adam/transformer_encoder_1/dense_90/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_274378


StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename-transformer_encoder_1/embedding_38/embeddingsdense_89/kerneldense_89/bias
Variable_2
Variable_1Variablelayer_normalization_26/gammalayer_normalization_26/beta%transformer_encoder_1/dense_90/kernel#transformer_encoder_1/dense_90/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcount4Adam/transformer_encoder_1/embedding_38/embeddings/mAdam/dense_89/kernel/mAdam/dense_89/bias/mAdam/Variable/m_2Adam/Variable/m_1Adam/Variable/m#Adam/layer_normalization_26/gamma/m"Adam/layer_normalization_26/beta/m,Adam/transformer_encoder_1/dense_90/kernel/m*Adam/transformer_encoder_1/dense_90/bias/m4Adam/transformer_encoder_1/embedding_38/embeddings/vAdam/dense_89/kernel/vAdam/dense_89/bias/vAdam/Variable/v_2Adam/Variable/v_1Adam/Variable/v#Adam/layer_normalization_26/gamma/v"Adam/layer_normalization_26/beta/v,Adam/transformer_encoder_1/dense_90/kernel/v*Adam/transformer_encoder_1/dense_90/bias/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_274505¯Æ	
È

)__inference_dense_90_layer_call_fn_274227

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_90_layer_call_and_return_conditional_losses_273664o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
Ï
Z__inference_transformer_block_rank_four_25_layer_call_and_return_conditional_losses_273634

inputs4
"attention_head_rank_four_25_273503:@@4
"attention_head_rank_four_25_273505:@@4
"attention_head_rank_four_25_273507:@@A
3layer_normalization_26_cast_readvariableop_resource:@C
5layer_normalization_26_cast_1_readvariableop_resource:@<
*dense_89_tensordot_readvariableop_resource:@@6
(dense_89_biasadd_readvariableop_resource:@
identity¢3attention_head_rank_four_25/StatefulPartitionedCall¢dense_89/BiasAdd/ReadVariableOp¢!dense_89/Tensordot/ReadVariableOp¢*layer_normalization_26/Cast/ReadVariableOp¢,layer_normalization_26/Cast_1/ReadVariableOp¢,layer_normalization_26/Cast_2/ReadVariableOp¢,layer_normalization_26/Cast_3/ReadVariableOp¿
3attention_head_rank_four_25/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputs"attention_head_rank_four_25_273503"attention_head_rank_four_25_273505"attention_head_rank_four_25_273507*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_273321
addAddV2inputs<attention_head_rank_four_25/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
layer_normalization_26/ShapeShapeadd:z:0*
T0*
_output_shapes
:t
*layer_normalization_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,layer_normalization_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,layer_normalization_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$layer_normalization_26/strided_sliceStridedSlice%layer_normalization_26/Shape:output:03layer_normalization_26/strided_slice/stack:output:05layer_normalization_26/strided_slice/stack_1:output:05layer_normalization_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
layer_normalization_26/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_26/mulMul%layer_normalization_26/mul/x:output:0-layer_normalization_26/strided_slice:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
&layer_normalization_26/strided_slice_1StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_1/stack:output:07layer_normalization_26/strided_slice_1/stack_1:output:07layer_normalization_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_26/mul_1Mullayer_normalization_26/mul:z:0/layer_normalization_26/strided_slice_1:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
&layer_normalization_26/strided_slice_2StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_2/stack:output:07layer_normalization_26/strided_slice_2/stack_1:output:07layer_normalization_26/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_26/mul_2Mul layer_normalization_26/mul_1:z:0/layer_normalization_26/strided_slice_2:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
&layer_normalization_26/strided_slice_3StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_3/stack:output:07layer_normalization_26/strided_slice_3/stack_1:output:07layer_normalization_26/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_26/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_26/mul_3Mul'layer_normalization_26/mul_3/x:output:0/layer_normalization_26/strided_slice_3:output:0*
T0*
_output_shapes
: h
&layer_normalization_26/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&layer_normalization_26/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
$layer_normalization_26/Reshape/shapePack/layer_normalization_26/Reshape/shape/0:output:0 layer_normalization_26/mul_2:z:0 layer_normalization_26/mul_3:z:0/layer_normalization_26/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_26/ReshapeReshapeadd:z:0-layer_normalization_26/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
"layer_normalization_26/ones/packedPack layer_normalization_26/mul_2:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_26/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ª
layer_normalization_26/onesFill+layer_normalization_26/ones/packed:output:0*layer_normalization_26/ones/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
#layer_normalization_26/zeros/packedPack layer_normalization_26/mul_2:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
layer_normalization_26/zerosFill,layer_normalization_26/zeros/packed:output:0+layer_normalization_26/zeros/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
layer_normalization_26/ConstConst*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_26/Const_1Const*
_output_shapes
: *
dtype0*
valueB ­
'layer_normalization_26/FusedBatchNormV3FusedBatchNormV3'layer_normalization_26/Reshape:output:0$layer_normalization_26/ones:output:0%layer_normalization_26/zeros:output:0%layer_normalization_26/Const:output:0'layer_normalization_26/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:*
data_formatNCHW*
epsilon%o:¹
 layer_normalization_26/Reshape_1Reshape+layer_normalization_26/FusedBatchNormV3:y:0%layer_normalization_26/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*layer_normalization_26/Cast/ReadVariableOpReadVariableOp3layer_normalization_26_cast_readvariableop_resource*
_output_shapes
:@*
dtype0¼
layer_normalization_26/mul_4Mul)layer_normalization_26/Reshape_1:output:02layer_normalization_26/Cast/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,layer_normalization_26/Cast_1/ReadVariableOpReadVariableOp5layer_normalization_26_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0µ
layer_normalization_26/addAddV2 layer_normalization_26/mul_4:z:04layer_normalization_26/Cast_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!dense_89/Tensordot/ReadVariableOpReadVariableOp*dense_89_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0a
dense_89/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:l
dense_89/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          f
dense_89/Tensordot/ShapeShapelayer_normalization_26/add:z:0*
T0*
_output_shapes
:b
 dense_89/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_89/Tensordot/GatherV2GatherV2!dense_89/Tensordot/Shape:output:0 dense_89/Tensordot/free:output:0)dense_89/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_89/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_89/Tensordot/GatherV2_1GatherV2!dense_89/Tensordot/Shape:output:0 dense_89/Tensordot/axes:output:0+dense_89/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_89/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_89/Tensordot/ProdProd$dense_89/Tensordot/GatherV2:output:0!dense_89/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_89/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_89/Tensordot/Prod_1Prod&dense_89/Tensordot/GatherV2_1:output:0#dense_89/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_89/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_89/Tensordot/concatConcatV2 dense_89/Tensordot/free:output:0 dense_89/Tensordot/axes:output:0'dense_89/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_89/Tensordot/stackPack dense_89/Tensordot/Prod:output:0"dense_89/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:§
dense_89/Tensordot/transpose	Transposelayer_normalization_26/add:z:0"dense_89/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
dense_89/Tensordot/ReshapeReshape dense_89/Tensordot/transpose:y:0!dense_89/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_89/Tensordot/MatMulMatMul#dense_89/Tensordot/Reshape:output:0)dense_89/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
dense_89/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@b
 dense_89/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_89/Tensordot/concat_1ConcatV2$dense_89/Tensordot/GatherV2:output:0#dense_89/Tensordot/Const_2:output:0)dense_89/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¢
dense_89/TensordotReshape#dense_89/Tensordot/MatMul:product:0$dense_89/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_89/BiasAdd/ReadVariableOpReadVariableOp(dense_89_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_89/BiasAddBiasAdddense_89/Tensordot:output:0'dense_89/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
add_1AddV2layer_normalization_26/add:z:0dense_89/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
layer_normalization_26/Shape_1Shape	add_1:z:0*
T0*
_output_shapes
:v
,layer_normalization_26/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.layer_normalization_26/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&layer_normalization_26/strided_slice_4StridedSlice'layer_normalization_26/Shape_1:output:05layer_normalization_26/strided_slice_4/stack:output:07layer_normalization_26/strided_slice_4/stack_1:output:07layer_normalization_26/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_26/mul_5/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_26/mul_5Mul'layer_normalization_26/mul_5/x:output:0/layer_normalization_26/strided_slice_4:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&layer_normalization_26/strided_slice_5StridedSlice'layer_normalization_26/Shape_1:output:05layer_normalization_26/strided_slice_5/stack:output:07layer_normalization_26/strided_slice_5/stack_1:output:07layer_normalization_26/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_26/mul_6Mul layer_normalization_26/mul_5:z:0/layer_normalization_26/strided_slice_5:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&layer_normalization_26/strided_slice_6StridedSlice'layer_normalization_26/Shape_1:output:05layer_normalization_26/strided_slice_6/stack:output:07layer_normalization_26/strided_slice_6/stack_1:output:07layer_normalization_26/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_26/mul_7Mul layer_normalization_26/mul_6:z:0/layer_normalization_26/strided_slice_6:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&layer_normalization_26/strided_slice_7StridedSlice'layer_normalization_26/Shape_1:output:05layer_normalization_26/strided_slice_7/stack:output:07layer_normalization_26/strided_slice_7/stack_1:output:07layer_normalization_26/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_26/mul_8/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_26/mul_8Mul'layer_normalization_26/mul_8/x:output:0/layer_normalization_26/strided_slice_7:output:0*
T0*
_output_shapes
: j
(layer_normalization_26/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :j
(layer_normalization_26/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
&layer_normalization_26/Reshape_2/shapePack1layer_normalization_26/Reshape_2/shape/0:output:0 layer_normalization_26/mul_7:z:0 layer_normalization_26/mul_8:z:01layer_normalization_26/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:¡
 layer_normalization_26/Reshape_2Reshape	add_1:z:0/layer_normalization_26/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
$layer_normalization_26/ones_1/packedPack layer_normalization_26/mul_7:z:0*
N*
T0*
_output_shapes
:h
#layer_normalization_26/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
layer_normalization_26/ones_1Fill-layer_normalization_26/ones_1/packed:output:0,layer_normalization_26/ones_1/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
%layer_normalization_26/zeros_1/packedPack layer_normalization_26/mul_7:z:0*
N*
T0*
_output_shapes
:i
$layer_normalization_26/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ³
layer_normalization_26/zeros_1Fill.layer_normalization_26/zeros_1/packed:output:0-layer_normalization_26/zeros_1/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
layer_normalization_26/Const_2Const*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_26/Const_3Const*
_output_shapes
: *
dtype0*
valueB ·
)layer_normalization_26/FusedBatchNormV3_1FusedBatchNormV3)layer_normalization_26/Reshape_2:output:0&layer_normalization_26/ones_1:output:0'layer_normalization_26/zeros_1:output:0'layer_normalization_26/Const_2:output:0'layer_normalization_26/Const_3:output:0*
T0*
U0*o
_output_shapes]
[:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:*
data_formatNCHW*
epsilon%o:½
 layer_normalization_26/Reshape_3Reshape-layer_normalization_26/FusedBatchNormV3_1:y:0'layer_normalization_26/Shape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,layer_normalization_26/Cast_2/ReadVariableOpReadVariableOp3layer_normalization_26_cast_readvariableop_resource*
_output_shapes
:@*
dtype0¾
layer_normalization_26/mul_9Mul)layer_normalization_26/Reshape_3:output:04layer_normalization_26/Cast_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,layer_normalization_26/Cast_3/ReadVariableOpReadVariableOp5layer_normalization_26_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0·
layer_normalization_26/add_1AddV2 layer_normalization_26/mul_9:z:04layer_normalization_26/Cast_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
ReluRelu layer_normalization_26/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ü
NoOpNoOp4^attention_head_rank_four_25/StatefulPartitionedCall ^dense_89/BiasAdd/ReadVariableOp"^dense_89/Tensordot/ReadVariableOp+^layer_normalization_26/Cast/ReadVariableOp-^layer_normalization_26/Cast_1/ReadVariableOp-^layer_normalization_26/Cast_2/ReadVariableOp-^layer_normalization_26/Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ@: : : : : : : 2j
3attention_head_rank_four_25/StatefulPartitionedCall3attention_head_rank_four_25/StatefulPartitionedCall2B
dense_89/BiasAdd/ReadVariableOpdense_89/BiasAdd/ReadVariableOp2F
!dense_89/Tensordot/ReadVariableOp!dense_89/Tensordot/ReadVariableOp2X
*layer_normalization_26/Cast/ReadVariableOp*layer_normalization_26/Cast/ReadVariableOp2\
,layer_normalization_26/Cast_1/ReadVariableOp,layer_normalization_26/Cast_1/ReadVariableOp2\
,layer_normalization_26/Cast_2/ReadVariableOp,layer_normalization_26/Cast_2/ReadVariableOp2\
,layer_normalization_26/Cast_3/ReadVariableOp,layer_normalization_26/Cast_3/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

ö
D__inference_dense_90_layer_call_and_return_conditional_losses_273664

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·I
þ
__inference_call_273321
inputs_for_keys
inputs_for_values
inputs_for_queries7
%einsum_einsum_readvariableop_resource:@@9
'einsum_1_einsum_readvariableop_resource:@@9
'einsum_2_einsum_readvariableop_resource:@@
identity¢einsum/Einsum/ReadVariableOp¢einsum_1/Einsum/ReadVariableOp¢einsum_2/Einsum/ReadVariableOp
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*
_output_shapes

:@@*
dtype0®
einsum/EinsumEinsuminputs_for_keys$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
equationixjk,kl->ixjl
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*
_output_shapes

:@@*
dtype0´
einsum_1/EinsumEinsuminputs_for_values&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
equationixjk,kl->ixjl
einsum_2/Einsum/ReadVariableOpReadVariableOp'einsum_2_einsum_readvariableop_resource*
_output_shapes

:@@*
dtype0µ
einsum_2/EinsumEinsuminputs_for_queries&einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
equationixjk,kl->ixjl
#attention_matrix_rank_four_25/ConstConst*"
_output_shapes
:*
dtype0*°
value¦B£"      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ          ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ              ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                          ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                              ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                          ÿ  ÿ  ÿ  ÿ  ÿ                                              ÿ  ÿ  ÿ  ÿ                                                  ÿ  ÿ  ÿ                                                      ÿ  ÿ                                                          ÿ                                                                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ          ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ              ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                          ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                              ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                          ÿ  ÿ  ÿ  ÿ  ÿ                                              ÿ  ÿ  ÿ  ÿ                                                  ÿ  ÿ  ÿ                                                      ÿ  ÿ                                                          ÿ                                                                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ          ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ              ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                          ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                              ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                          ÿ  ÿ  ÿ  ÿ  ÿ                                              ÿ  ÿ  ÿ  ÿ                                                  ÿ  ÿ  ÿ                                                      ÿ  ÿ                                                          ÿ                                                                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ          ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ              ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                          ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                              ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                          ÿ  ÿ  ÿ  ÿ  ÿ                                              ÿ  ÿ  ÿ  ÿ                                                  ÿ  ÿ  ÿ                                                      ÿ  ÿ                                                          ÿ                                                            
+attention_matrix_rank_four_25/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         Å
%attention_matrix_rank_four_25/ReshapeReshape,attention_matrix_rank_four_25/Const:output:04attention_matrix_rank_four_25/Reshape/shape:output:0*
T0*&
_output_shapes
:i
#attention_matrix_rank_four_25/ShapeShapeeinsum/Einsum:output:0*
T0*
_output_shapes
:{
1attention_matrix_rank_four_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3attention_matrix_rank_four_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3attention_matrix_rank_four_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+attention_matrix_rank_four_25/strided_sliceStridedSlice,attention_matrix_rank_four_25/Shape:output:0:attention_matrix_rank_four_25/strided_slice/stack:output:0<attention_matrix_rank_four_25/strided_slice/stack_1:output:0<attention_matrix_rank_four_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.attention_matrix_rank_four_25/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :p
.attention_matrix_rank_four_25/Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :p
.attention_matrix_rank_four_25/Tile/multiples/3Const*
_output_shapes
: *
dtype0*
value	B :Ã
,attention_matrix_rank_four_25/Tile/multiplesPack4attention_matrix_rank_four_25/strided_slice:output:07attention_matrix_rank_four_25/Tile/multiples/1:output:07attention_matrix_rank_four_25/Tile/multiples/2:output:07attention_matrix_rank_four_25/Tile/multiples/3:output:0*
N*
T0*
_output_shapes
:Ë
"attention_matrix_rank_four_25/TileTile.attention_matrix_rank_four_25/Reshape:output:05attention_matrix_rank_four_25/Tile/multiples:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,attention_matrix_rank_four_25/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ½
'attention_matrix_rank_four_25/transpose	Transposeeinsum/Einsum:output:05attention_matrix_rank_four_25/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
$attention_matrix_rank_four_25/MatMulBatchMatMulV2einsum_2/Einsum:output:0+attention_matrix_rank_four_25/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
'attention_matrix_rank_four_25/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *öÞw@Ë
%attention_matrix_rank_four_25/truedivRealDiv-attention_matrix_rank_four_25/MatMul:output:00attention_matrix_rank_four_25/truediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
!attention_matrix_rank_four_25/AddAddV2)attention_matrix_rank_four_25/truediv:z:0+attention_matrix_rank_four_25/Tile:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%attention_matrix_rank_four_25/SoftmaxSoftmax%attention_matrix_rank_four_25/Add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
MatMulBatchMatMulV2/attention_matrix_rank_four_25/Softmax:softmax:0einsum_1/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
IdentityIdentityMatMul:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
NoOpNoOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_2/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : 2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_2/Einsum/ReadVariableOpeinsum_2/Einsum/ReadVariableOp:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)
_user_specified_nameinputs_for_keys:b^
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+
_user_specified_nameinputs_for_values:c_
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,
_user_specified_nameinputs_for_queries
ä
¦
H__inference_embedding_38_layer_call_and_return_conditional_losses_273496

inputs*
embedding_lookup_273490:	'@
identity¢embedding_lookup½
embedding_lookupResourceGatherembedding_lookup_273490inputs*
Tindices0**
_class 
loc:@embedding_lookup/273490*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0¦
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/273490*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç)
Á
!__inference__wrapped_model_273480
input_1M
:transformer_encoder_1_embedding_38_embedding_lookup_273279:	'@M
;transformer_encoder_1_transformer_block_rank_four_25_273454:@@M
;transformer_encoder_1_transformer_block_rank_four_25_273456:@@M
;transformer_encoder_1_transformer_block_rank_four_25_273458:@@I
;transformer_encoder_1_transformer_block_rank_four_25_273460:@I
;transformer_encoder_1_transformer_block_rank_four_25_273462:@M
;transformer_encoder_1_transformer_block_rank_four_25_273464:@@I
;transformer_encoder_1_transformer_block_rank_four_25_273466:@P
=transformer_encoder_1_dense_90_matmul_readvariableop_resource:	L
>transformer_encoder_1_dense_90_biasadd_readvariableop_resource:
identity¢5transformer_encoder_1/dense_90/BiasAdd/ReadVariableOp¢4transformer_encoder_1/dense_90/MatMul/ReadVariableOp¢3transformer_encoder_1/embedding_38/embedding_lookup¢Ltransformer_encoder_1/transformer_block_rank_four_25/StatefulPartitionedCall§
3transformer_encoder_1/embedding_38/embedding_lookupResourceGather:transformer_encoder_1_embedding_38_embedding_lookup_273279input_1*
Tindices0*M
_classC
A?loc:@transformer_encoder_1/embedding_38/embedding_lookup/273279*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0
<transformer_encoder_1/embedding_38/embedding_lookup/IdentityIdentity<transformer_encoder_1/embedding_38/embedding_lookup:output:0*
T0*M
_classC
A?loc:@transformer_encoder_1/embedding_38/embedding_lookup/273279*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ë
>transformer_encoder_1/embedding_38/embedding_lookup/Identity_1IdentityEtransformer_encoder_1/embedding_38/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Î
Ltransformer_encoder_1/transformer_block_rank_four_25/StatefulPartitionedCallStatefulPartitionedCallGtransformer_encoder_1/embedding_38/embedding_lookup/Identity_1:output:0;transformer_encoder_1_transformer_block_rank_four_25_273454;transformer_encoder_1_transformer_block_rank_four_25_273456;transformer_encoder_1_transformer_block_rank_four_25_273458;transformer_encoder_1_transformer_block_rank_four_25_273460;transformer_encoder_1_transformer_block_rank_four_25_273462;transformer_encoder_1_transformer_block_rank_four_25_273464;transformer_encoder_1_transformer_block_rank_four_25_273466*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_273453t
#transformer_encoder_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   è
%transformer_encoder_1/flatten/ReshapeReshapeUtransformer_encoder_1/transformer_block_rank_four_25/StatefulPartitionedCall:output:0,transformer_encoder_1/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&transformer_encoder_1/dropout/IdentityIdentity.transformer_encoder_1/flatten/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
4transformer_encoder_1/dense_90/MatMul/ReadVariableOpReadVariableOp=transformer_encoder_1_dense_90_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ð
%transformer_encoder_1/dense_90/MatMulMatMul/transformer_encoder_1/dropout/Identity:output:0<transformer_encoder_1/dense_90/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
5transformer_encoder_1/dense_90/BiasAdd/ReadVariableOpReadVariableOp>transformer_encoder_1_dense_90_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ó
&transformer_encoder_1/dense_90/BiasAddBiasAdd/transformer_encoder_1/dense_90/MatMul:product:0=transformer_encoder_1/dense_90/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&transformer_encoder_1/dense_90/SoftmaxSoftmax/transformer_encoder_1/dense_90/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity0transformer_encoder_1/dense_90/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp6^transformer_encoder_1/dense_90/BiasAdd/ReadVariableOp5^transformer_encoder_1/dense_90/MatMul/ReadVariableOp4^transformer_encoder_1/embedding_38/embedding_lookupM^transformer_encoder_1/transformer_block_rank_four_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2n
5transformer_encoder_1/dense_90/BiasAdd/ReadVariableOp5transformer_encoder_1/dense_90/BiasAdd/ReadVariableOp2l
4transformer_encoder_1/dense_90/MatMul/ReadVariableOp4transformer_encoder_1/dense_90/MatMul/ReadVariableOp2j
3transformer_encoder_1/embedding_38/embedding_lookup3transformer_encoder_1/embedding_38/embedding_lookup2
Ltransformer_encoder_1/transformer_block_rank_four_25/StatefulPartitionedCallLtransformer_encoder_1/transformer_block_rank_four_25/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
©
À
Q__inference_transformer_encoder_1_layer_call_and_return_conditional_losses_273879

transcript7
$embedding_38_embedding_lookup_273848:	'@7
%transformer_block_rank_four_25_273853:@@7
%transformer_block_rank_four_25_273855:@@7
%transformer_block_rank_four_25_273857:@@3
%transformer_block_rank_four_25_273859:@3
%transformer_block_rank_four_25_273861:@7
%transformer_block_rank_four_25_273863:@@3
%transformer_block_rank_four_25_273865:@:
'dense_90_matmul_readvariableop_resource:	6
(dense_90_biasadd_readvariableop_resource:
identity¢dense_90/BiasAdd/ReadVariableOp¢dense_90/MatMul/ReadVariableOp¢embedding_38/embedding_lookup¢6transformer_block_rank_four_25/StatefulPartitionedCallè
embedding_38/embedding_lookupResourceGather$embedding_38_embedding_lookup_273848
transcript*
Tindices0*7
_class-
+)loc:@embedding_38/embedding_lookup/273848*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0Í
&embedding_38/embedding_lookup/IdentityIdentity&embedding_38/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_38/embedding_lookup/273848*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(embedding_38/embedding_lookup/Identity_1Identity/embedding_38/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
6transformer_block_rank_four_25/StatefulPartitionedCallStatefulPartitionedCall1embedding_38/embedding_lookup/Identity_1:output:0%transformer_block_rank_four_25_273853%transformer_block_rank_four_25_273855%transformer_block_rank_four_25_273857%transformer_block_rank_four_25_273859%transformer_block_rank_four_25_273861%transformer_block_rank_four_25_273863%transformer_block_rank_four_25_273865*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_273453^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
flatten/ReshapeReshape?transformer_block_rank_four_25/StatefulPartitionedCall:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_90/MatMul/ReadVariableOpReadVariableOp'dense_90_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_90/MatMulMatMuldropout/Identity:output:0&dense_90/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_90/BiasAdd/ReadVariableOpReadVariableOp(dense_90_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_90/BiasAddBiasAdddense_90/MatMul:product:0'dense_90/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_90/SoftmaxSoftmaxdense_90/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_90/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ
NoOpNoOp ^dense_90/BiasAdd/ReadVariableOp^dense_90/MatMul/ReadVariableOp^embedding_38/embedding_lookup7^transformer_block_rank_four_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2B
dense_90/BiasAdd/ReadVariableOpdense_90/BiasAdd/ReadVariableOp2@
dense_90/MatMul/ReadVariableOpdense_90/MatMul/ReadVariableOp2>
embedding_38/embedding_lookupembedding_38/embedding_lookup2p
6transformer_block_rank_four_25/StatefulPartitionedCall6transformer_block_rank_four_25/StatefulPartitionedCall:W S
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
transcript
§
¼
"__inference__traced_restore_274505
file_prefixQ
>assignvariableop_transformer_encoder_1_embedding_38_embeddings:	'@4
"assignvariableop_1_dense_89_kernel:@@.
 assignvariableop_2_dense_89_bias:@/
assignvariableop_3_variable_2:@@/
assignvariableop_4_variable_1:@@-
assignvariableop_5_variable:@@=
/assignvariableop_6_layer_normalization_26_gamma:@<
.assignvariableop_7_layer_normalization_26_beta:@K
8assignvariableop_8_transformer_encoder_1_dense_90_kernel:	D
6assignvariableop_9_transformer_encoder_1_dense_90_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: #
assignvariableop_17_total: #
assignvariableop_18_count: [
Hassignvariableop_19_adam_transformer_encoder_1_embedding_38_embeddings_m:	'@<
*assignvariableop_20_adam_dense_89_kernel_m:@@6
(assignvariableop_21_adam_dense_89_bias_m:@7
%assignvariableop_22_adam_variable_m_2:@@7
%assignvariableop_23_adam_variable_m_1:@@5
#assignvariableop_24_adam_variable_m:@@E
7assignvariableop_25_adam_layer_normalization_26_gamma_m:@D
6assignvariableop_26_adam_layer_normalization_26_beta_m:@S
@assignvariableop_27_adam_transformer_encoder_1_dense_90_kernel_m:	L
>assignvariableop_28_adam_transformer_encoder_1_dense_90_bias_m:[
Hassignvariableop_29_adam_transformer_encoder_1_embedding_38_embeddings_v:	'@<
*assignvariableop_30_adam_dense_89_kernel_v:@@6
(assignvariableop_31_adam_dense_89_bias_v:@7
%assignvariableop_32_adam_variable_v_2:@@7
%assignvariableop_33_adam_variable_v_1:@@5
#assignvariableop_34_adam_variable_v:@@E
7assignvariableop_35_adam_layer_normalization_26_gamma_v:@D
6assignvariableop_36_adam_layer_normalization_26_beta_v:@S
@assignvariableop_37_adam_transformer_encoder_1_dense_90_kernel_v:	L
>assignvariableop_38_adam_transformer_encoder_1_dense_90_bias_v:
identity_40¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ª
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*Ð
valueÆBÃ(B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÀ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B é
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¶
_output_shapes£
 ::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOpAssignVariableOp>assignvariableop_transformer_encoder_1_embedding_38_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_89_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense_89_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_2Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_variableIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp/assignvariableop_6_layer_normalization_26_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp.assignvariableop_7_layer_normalization_26_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_8AssignVariableOp8assignvariableop_8_transformer_encoder_1_dense_90_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_9AssignVariableOp6assignvariableop_9_transformer_encoder_1_dense_90_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¹
AssignVariableOp_19AssignVariableOpHassignvariableop_19_adam_transformer_encoder_1_embedding_38_embeddings_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_89_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_89_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_variable_m_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp%assignvariableop_23_adam_variable_m_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp#assignvariableop_24_adam_variable_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adam_layer_normalization_26_gamma_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_layer_normalization_26_beta_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_transformer_encoder_1_dense_90_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_28AssignVariableOp>assignvariableop_28_adam_transformer_encoder_1_dense_90_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¹
AssignVariableOp_29AssignVariableOpHassignvariableop_29_adam_transformer_encoder_1_embedding_38_embeddings_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_dense_89_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_dense_89_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_variable_v_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp%assignvariableop_33_adam_variable_v_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp#assignvariableop_34_adam_variable_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_35AssignVariableOp7assignvariableop_35_adam_layer_normalization_26_gamma_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_36AssignVariableOp6assignvariableop_36_adam_layer_normalization_26_beta_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_37AssignVariableOp@assignvariableop_37_adam_transformer_encoder_1_dense_90_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_38AssignVariableOp>assignvariableop_38_adam_transformer_encoder_1_dense_90_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ©
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
É
Ï
Z__inference_transformer_block_rank_four_25_layer_call_and_return_conditional_losses_274182

inputs4
"attention_head_rank_four_25_274051:@@4
"attention_head_rank_four_25_274053:@@4
"attention_head_rank_four_25_274055:@@A
3layer_normalization_26_cast_readvariableop_resource:@C
5layer_normalization_26_cast_1_readvariableop_resource:@<
*dense_89_tensordot_readvariableop_resource:@@6
(dense_89_biasadd_readvariableop_resource:@
identity¢3attention_head_rank_four_25/StatefulPartitionedCall¢dense_89/BiasAdd/ReadVariableOp¢!dense_89/Tensordot/ReadVariableOp¢*layer_normalization_26/Cast/ReadVariableOp¢,layer_normalization_26/Cast_1/ReadVariableOp¢,layer_normalization_26/Cast_2/ReadVariableOp¢,layer_normalization_26/Cast_3/ReadVariableOp¿
3attention_head_rank_four_25/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputs"attention_head_rank_four_25_274051"attention_head_rank_four_25_274053"attention_head_rank_four_25_274055*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_273321
addAddV2inputs<attention_head_rank_four_25/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
layer_normalization_26/ShapeShapeadd:z:0*
T0*
_output_shapes
:t
*layer_normalization_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,layer_normalization_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,layer_normalization_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$layer_normalization_26/strided_sliceStridedSlice%layer_normalization_26/Shape:output:03layer_normalization_26/strided_slice/stack:output:05layer_normalization_26/strided_slice/stack_1:output:05layer_normalization_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
layer_normalization_26/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_26/mulMul%layer_normalization_26/mul/x:output:0-layer_normalization_26/strided_slice:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
&layer_normalization_26/strided_slice_1StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_1/stack:output:07layer_normalization_26/strided_slice_1/stack_1:output:07layer_normalization_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_26/mul_1Mullayer_normalization_26/mul:z:0/layer_normalization_26/strided_slice_1:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
&layer_normalization_26/strided_slice_2StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_2/stack:output:07layer_normalization_26/strided_slice_2/stack_1:output:07layer_normalization_26/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_26/mul_2Mul layer_normalization_26/mul_1:z:0/layer_normalization_26/strided_slice_2:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
&layer_normalization_26/strided_slice_3StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_3/stack:output:07layer_normalization_26/strided_slice_3/stack_1:output:07layer_normalization_26/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_26/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_26/mul_3Mul'layer_normalization_26/mul_3/x:output:0/layer_normalization_26/strided_slice_3:output:0*
T0*
_output_shapes
: h
&layer_normalization_26/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&layer_normalization_26/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
$layer_normalization_26/Reshape/shapePack/layer_normalization_26/Reshape/shape/0:output:0 layer_normalization_26/mul_2:z:0 layer_normalization_26/mul_3:z:0/layer_normalization_26/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_26/ReshapeReshapeadd:z:0-layer_normalization_26/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
"layer_normalization_26/ones/packedPack layer_normalization_26/mul_2:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_26/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ª
layer_normalization_26/onesFill+layer_normalization_26/ones/packed:output:0*layer_normalization_26/ones/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
#layer_normalization_26/zeros/packedPack layer_normalization_26/mul_2:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
layer_normalization_26/zerosFill,layer_normalization_26/zeros/packed:output:0+layer_normalization_26/zeros/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
layer_normalization_26/ConstConst*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_26/Const_1Const*
_output_shapes
: *
dtype0*
valueB ­
'layer_normalization_26/FusedBatchNormV3FusedBatchNormV3'layer_normalization_26/Reshape:output:0$layer_normalization_26/ones:output:0%layer_normalization_26/zeros:output:0%layer_normalization_26/Const:output:0'layer_normalization_26/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:*
data_formatNCHW*
epsilon%o:¹
 layer_normalization_26/Reshape_1Reshape+layer_normalization_26/FusedBatchNormV3:y:0%layer_normalization_26/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*layer_normalization_26/Cast/ReadVariableOpReadVariableOp3layer_normalization_26_cast_readvariableop_resource*
_output_shapes
:@*
dtype0¼
layer_normalization_26/mul_4Mul)layer_normalization_26/Reshape_1:output:02layer_normalization_26/Cast/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,layer_normalization_26/Cast_1/ReadVariableOpReadVariableOp5layer_normalization_26_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0µ
layer_normalization_26/addAddV2 layer_normalization_26/mul_4:z:04layer_normalization_26/Cast_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!dense_89/Tensordot/ReadVariableOpReadVariableOp*dense_89_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0a
dense_89/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:l
dense_89/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          f
dense_89/Tensordot/ShapeShapelayer_normalization_26/add:z:0*
T0*
_output_shapes
:b
 dense_89/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_89/Tensordot/GatherV2GatherV2!dense_89/Tensordot/Shape:output:0 dense_89/Tensordot/free:output:0)dense_89/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_89/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_89/Tensordot/GatherV2_1GatherV2!dense_89/Tensordot/Shape:output:0 dense_89/Tensordot/axes:output:0+dense_89/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_89/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_89/Tensordot/ProdProd$dense_89/Tensordot/GatherV2:output:0!dense_89/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_89/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_89/Tensordot/Prod_1Prod&dense_89/Tensordot/GatherV2_1:output:0#dense_89/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_89/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_89/Tensordot/concatConcatV2 dense_89/Tensordot/free:output:0 dense_89/Tensordot/axes:output:0'dense_89/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_89/Tensordot/stackPack dense_89/Tensordot/Prod:output:0"dense_89/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:§
dense_89/Tensordot/transpose	Transposelayer_normalization_26/add:z:0"dense_89/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
dense_89/Tensordot/ReshapeReshape dense_89/Tensordot/transpose:y:0!dense_89/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_89/Tensordot/MatMulMatMul#dense_89/Tensordot/Reshape:output:0)dense_89/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
dense_89/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@b
 dense_89/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_89/Tensordot/concat_1ConcatV2$dense_89/Tensordot/GatherV2:output:0#dense_89/Tensordot/Const_2:output:0)dense_89/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¢
dense_89/TensordotReshape#dense_89/Tensordot/MatMul:product:0$dense_89/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_89/BiasAdd/ReadVariableOpReadVariableOp(dense_89_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_89/BiasAddBiasAdddense_89/Tensordot:output:0'dense_89/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
add_1AddV2layer_normalization_26/add:z:0dense_89/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
layer_normalization_26/Shape_1Shape	add_1:z:0*
T0*
_output_shapes
:v
,layer_normalization_26/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.layer_normalization_26/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&layer_normalization_26/strided_slice_4StridedSlice'layer_normalization_26/Shape_1:output:05layer_normalization_26/strided_slice_4/stack:output:07layer_normalization_26/strided_slice_4/stack_1:output:07layer_normalization_26/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_26/mul_5/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_26/mul_5Mul'layer_normalization_26/mul_5/x:output:0/layer_normalization_26/strided_slice_4:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&layer_normalization_26/strided_slice_5StridedSlice'layer_normalization_26/Shape_1:output:05layer_normalization_26/strided_slice_5/stack:output:07layer_normalization_26/strided_slice_5/stack_1:output:07layer_normalization_26/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_26/mul_6Mul layer_normalization_26/mul_5:z:0/layer_normalization_26/strided_slice_5:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&layer_normalization_26/strided_slice_6StridedSlice'layer_normalization_26/Shape_1:output:05layer_normalization_26/strided_slice_6/stack:output:07layer_normalization_26/strided_slice_6/stack_1:output:07layer_normalization_26/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_26/mul_7Mul layer_normalization_26/mul_6:z:0/layer_normalization_26/strided_slice_6:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&layer_normalization_26/strided_slice_7StridedSlice'layer_normalization_26/Shape_1:output:05layer_normalization_26/strided_slice_7/stack:output:07layer_normalization_26/strided_slice_7/stack_1:output:07layer_normalization_26/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_26/mul_8/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_26/mul_8Mul'layer_normalization_26/mul_8/x:output:0/layer_normalization_26/strided_slice_7:output:0*
T0*
_output_shapes
: j
(layer_normalization_26/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :j
(layer_normalization_26/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
&layer_normalization_26/Reshape_2/shapePack1layer_normalization_26/Reshape_2/shape/0:output:0 layer_normalization_26/mul_7:z:0 layer_normalization_26/mul_8:z:01layer_normalization_26/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:¡
 layer_normalization_26/Reshape_2Reshape	add_1:z:0/layer_normalization_26/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
$layer_normalization_26/ones_1/packedPack layer_normalization_26/mul_7:z:0*
N*
T0*
_output_shapes
:h
#layer_normalization_26/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
layer_normalization_26/ones_1Fill-layer_normalization_26/ones_1/packed:output:0,layer_normalization_26/ones_1/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
%layer_normalization_26/zeros_1/packedPack layer_normalization_26/mul_7:z:0*
N*
T0*
_output_shapes
:i
$layer_normalization_26/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ³
layer_normalization_26/zeros_1Fill.layer_normalization_26/zeros_1/packed:output:0-layer_normalization_26/zeros_1/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
layer_normalization_26/Const_2Const*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_26/Const_3Const*
_output_shapes
: *
dtype0*
valueB ·
)layer_normalization_26/FusedBatchNormV3_1FusedBatchNormV3)layer_normalization_26/Reshape_2:output:0&layer_normalization_26/ones_1:output:0'layer_normalization_26/zeros_1:output:0'layer_normalization_26/Const_2:output:0'layer_normalization_26/Const_3:output:0*
T0*
U0*o
_output_shapes]
[:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:*
data_formatNCHW*
epsilon%o:½
 layer_normalization_26/Reshape_3Reshape-layer_normalization_26/FusedBatchNormV3_1:y:0'layer_normalization_26/Shape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,layer_normalization_26/Cast_2/ReadVariableOpReadVariableOp3layer_normalization_26_cast_readvariableop_resource*
_output_shapes
:@*
dtype0¾
layer_normalization_26/mul_9Mul)layer_normalization_26/Reshape_3:output:04layer_normalization_26/Cast_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,layer_normalization_26/Cast_3/ReadVariableOpReadVariableOp5layer_normalization_26_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0·
layer_normalization_26/add_1AddV2 layer_normalization_26/mul_9:z:04layer_normalization_26/Cast_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
ReluRelu layer_normalization_26/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ü
NoOpNoOp4^attention_head_rank_four_25/StatefulPartitionedCall ^dense_89/BiasAdd/ReadVariableOp"^dense_89/Tensordot/ReadVariableOp+^layer_normalization_26/Cast/ReadVariableOp-^layer_normalization_26/Cast_1/ReadVariableOp-^layer_normalization_26/Cast_2/ReadVariableOp-^layer_normalization_26/Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ@: : : : : : : 2j
3attention_head_rank_four_25/StatefulPartitionedCall3attention_head_rank_four_25/StatefulPartitionedCall2B
dense_89/BiasAdd/ReadVariableOpdense_89/BiasAdd/ReadVariableOp2F
!dense_89/Tensordot/ReadVariableOp!dense_89/Tensordot/ReadVariableOp2X
*layer_normalization_26/Cast/ReadVariableOp*layer_normalization_26/Cast/ReadVariableOp2\
,layer_normalization_26/Cast_1/ReadVariableOp,layer_normalization_26/Cast_1/ReadVariableOp2\
,layer_normalization_26/Cast_2/ReadVariableOp,layer_normalization_26/Cast_2/ReadVariableOp2\
,layer_normalization_26/Cast_3/ReadVariableOp,layer_normalization_26/Cast_3/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ä
¦
H__inference_embedding_38_layer_call_and_return_conditional_losses_274029

inputs*
embedding_lookup_274023:	'@
identity¢embedding_lookup½
embedding_lookupResourceGatherembedding_lookup_274023inputs*
Tindices0**
_class 
loc:@embedding_lookup/274023*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0¦
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/274023*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

-__inference_embedding_38_layer_call_fn_274020

inputs
unknown:	'@
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_embedding_38_layer_call_and_return_conditional_losses_273496w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À


6__inference_transformer_encoder_1_layer_call_fn_273694
input_1
unknown:	'@
	unknown_0:@@
	unknown_1:@@
	unknown_2:@@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:	
	unknown_8:
identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_transformer_encoder_1_layer_call_and_return_conditional_losses_273671o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ï
â
Q__inference_transformer_encoder_1_layer_call_and_return_conditional_losses_273787
input_1&
embedding_38_273760:	'@7
%transformer_block_rank_four_25_273763:@@7
%transformer_block_rank_four_25_273765:@@7
%transformer_block_rank_four_25_273767:@@3
%transformer_block_rank_four_25_273769:@3
%transformer_block_rank_four_25_273771:@7
%transformer_block_rank_four_25_273773:@@3
%transformer_block_rank_four_25_273775:@"
dense_90_273781:	
dense_90_273783:
identity¢ dense_90/StatefulPartitionedCall¢$embedding_38/StatefulPartitionedCall¢6transformer_block_rank_four_25/StatefulPartitionedCallõ
$embedding_38/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_38_273760*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_embedding_38_layer_call_and_return_conditional_losses_273496Ç
6transformer_block_rank_four_25/StatefulPartitionedCallStatefulPartitionedCall-embedding_38/StatefulPartitionedCall:output:0%transformer_block_rank_four_25_273763%transformer_block_rank_four_25_273765%transformer_block_rank_four_25_273767%transformer_block_rank_four_25_273769%transformer_block_rank_four_25_273771%transformer_block_rank_four_25_273773%transformer_block_rank_four_25_273775*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_transformer_block_rank_four_25_layer_call_and_return_conditional_losses_273634^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
flatten/ReshapeReshape?transformer_block_rank_four_25/StatefulPartitionedCall:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_90/StatefulPartitionedCallStatefulPartitionedCalldropout/Identity:output:0dense_90_273781dense_90_273783*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_90_layer_call_and_return_conditional_losses_273664x
IdentityIdentity)dense_90/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
NoOpNoOp!^dense_90/StatefulPartitionedCall%^embedding_38/StatefulPartitionedCall7^transformer_block_rank_four_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2D
 dense_90/StatefulPartitionedCall dense_90/StatefulPartitionedCall2L
$embedding_38/StatefulPartitionedCall$embedding_38/StatefulPartitionedCall2p
6transformer_block_rank_four_25/StatefulPartitionedCall6transformer_block_rank_four_25/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
å	
µ
?__inference_transformer_block_rank_four_25_layer_call_fn_274048

inputs
unknown:@@
	unknown_0:@@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@@
	unknown_5:@
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_transformer_block_rank_four_25_layer_call_and_return_conditional_losses_273634w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ@: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

ö
D__inference_dense_90_layer_call_and_return_conditional_losses_274238

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


__inference_call_273453

inputs4
"attention_head_rank_four_25_273322:@@4
"attention_head_rank_four_25_273324:@@4
"attention_head_rank_four_25_273326:@@A
3layer_normalization_26_cast_readvariableop_resource:@C
5layer_normalization_26_cast_1_readvariableop_resource:@<
*dense_89_tensordot_readvariableop_resource:@@6
(dense_89_biasadd_readvariableop_resource:@
identity¢3attention_head_rank_four_25/StatefulPartitionedCall¢dense_89/BiasAdd/ReadVariableOp¢!dense_89/Tensordot/ReadVariableOp¢*layer_normalization_26/Cast/ReadVariableOp¢,layer_normalization_26/Cast_1/ReadVariableOp¢,layer_normalization_26/Cast_2/ReadVariableOp¢,layer_normalization_26/Cast_3/ReadVariableOp¿
3attention_head_rank_four_25/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputs"attention_head_rank_four_25_273322"attention_head_rank_four_25_273324"attention_head_rank_four_25_273326*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_273321
addAddV2inputs<attention_head_rank_four_25/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
layer_normalization_26/ShapeShapeadd:z:0*
T0*
_output_shapes
:t
*layer_normalization_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,layer_normalization_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,layer_normalization_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$layer_normalization_26/strided_sliceStridedSlice%layer_normalization_26/Shape:output:03layer_normalization_26/strided_slice/stack:output:05layer_normalization_26/strided_slice/stack_1:output:05layer_normalization_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
layer_normalization_26/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_26/mulMul%layer_normalization_26/mul/x:output:0-layer_normalization_26/strided_slice:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
&layer_normalization_26/strided_slice_1StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_1/stack:output:07layer_normalization_26/strided_slice_1/stack_1:output:07layer_normalization_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_26/mul_1Mullayer_normalization_26/mul:z:0/layer_normalization_26/strided_slice_1:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
&layer_normalization_26/strided_slice_2StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_2/stack:output:07layer_normalization_26/strided_slice_2/stack_1:output:07layer_normalization_26/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_26/mul_2Mul layer_normalization_26/mul_1:z:0/layer_normalization_26/strided_slice_2:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
&layer_normalization_26/strided_slice_3StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_3/stack:output:07layer_normalization_26/strided_slice_3/stack_1:output:07layer_normalization_26/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_26/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_26/mul_3Mul'layer_normalization_26/mul_3/x:output:0/layer_normalization_26/strided_slice_3:output:0*
T0*
_output_shapes
: h
&layer_normalization_26/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&layer_normalization_26/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
$layer_normalization_26/Reshape/shapePack/layer_normalization_26/Reshape/shape/0:output:0 layer_normalization_26/mul_2:z:0 layer_normalization_26/mul_3:z:0/layer_normalization_26/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_26/ReshapeReshapeadd:z:0-layer_normalization_26/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
"layer_normalization_26/ones/packedPack layer_normalization_26/mul_2:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_26/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ª
layer_normalization_26/onesFill+layer_normalization_26/ones/packed:output:0*layer_normalization_26/ones/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
#layer_normalization_26/zeros/packedPack layer_normalization_26/mul_2:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
layer_normalization_26/zerosFill,layer_normalization_26/zeros/packed:output:0+layer_normalization_26/zeros/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
layer_normalization_26/ConstConst*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_26/Const_1Const*
_output_shapes
: *
dtype0*
valueB ­
'layer_normalization_26/FusedBatchNormV3FusedBatchNormV3'layer_normalization_26/Reshape:output:0$layer_normalization_26/ones:output:0%layer_normalization_26/zeros:output:0%layer_normalization_26/Const:output:0'layer_normalization_26/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:*
data_formatNCHW*
epsilon%o:¹
 layer_normalization_26/Reshape_1Reshape+layer_normalization_26/FusedBatchNormV3:y:0%layer_normalization_26/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*layer_normalization_26/Cast/ReadVariableOpReadVariableOp3layer_normalization_26_cast_readvariableop_resource*
_output_shapes
:@*
dtype0¼
layer_normalization_26/mul_4Mul)layer_normalization_26/Reshape_1:output:02layer_normalization_26/Cast/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,layer_normalization_26/Cast_1/ReadVariableOpReadVariableOp5layer_normalization_26_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0µ
layer_normalization_26/addAddV2 layer_normalization_26/mul_4:z:04layer_normalization_26/Cast_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!dense_89/Tensordot/ReadVariableOpReadVariableOp*dense_89_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0a
dense_89/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:l
dense_89/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          f
dense_89/Tensordot/ShapeShapelayer_normalization_26/add:z:0*
T0*
_output_shapes
:b
 dense_89/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_89/Tensordot/GatherV2GatherV2!dense_89/Tensordot/Shape:output:0 dense_89/Tensordot/free:output:0)dense_89/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_89/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_89/Tensordot/GatherV2_1GatherV2!dense_89/Tensordot/Shape:output:0 dense_89/Tensordot/axes:output:0+dense_89/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_89/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_89/Tensordot/ProdProd$dense_89/Tensordot/GatherV2:output:0!dense_89/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_89/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_89/Tensordot/Prod_1Prod&dense_89/Tensordot/GatherV2_1:output:0#dense_89/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_89/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_89/Tensordot/concatConcatV2 dense_89/Tensordot/free:output:0 dense_89/Tensordot/axes:output:0'dense_89/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_89/Tensordot/stackPack dense_89/Tensordot/Prod:output:0"dense_89/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:§
dense_89/Tensordot/transpose	Transposelayer_normalization_26/add:z:0"dense_89/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
dense_89/Tensordot/ReshapeReshape dense_89/Tensordot/transpose:y:0!dense_89/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_89/Tensordot/MatMulMatMul#dense_89/Tensordot/Reshape:output:0)dense_89/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
dense_89/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@b
 dense_89/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_89/Tensordot/concat_1ConcatV2$dense_89/Tensordot/GatherV2:output:0#dense_89/Tensordot/Const_2:output:0)dense_89/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¢
dense_89/TensordotReshape#dense_89/Tensordot/MatMul:product:0$dense_89/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_89/BiasAdd/ReadVariableOpReadVariableOp(dense_89_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_89/BiasAddBiasAdddense_89/Tensordot:output:0'dense_89/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
add_1AddV2layer_normalization_26/add:z:0dense_89/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
layer_normalization_26/Shape_1Shape	add_1:z:0*
T0*
_output_shapes
:v
,layer_normalization_26/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.layer_normalization_26/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&layer_normalization_26/strided_slice_4StridedSlice'layer_normalization_26/Shape_1:output:05layer_normalization_26/strided_slice_4/stack:output:07layer_normalization_26/strided_slice_4/stack_1:output:07layer_normalization_26/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_26/mul_5/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_26/mul_5Mul'layer_normalization_26/mul_5/x:output:0/layer_normalization_26/strided_slice_4:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&layer_normalization_26/strided_slice_5StridedSlice'layer_normalization_26/Shape_1:output:05layer_normalization_26/strided_slice_5/stack:output:07layer_normalization_26/strided_slice_5/stack_1:output:07layer_normalization_26/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_26/mul_6Mul layer_normalization_26/mul_5:z:0/layer_normalization_26/strided_slice_5:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&layer_normalization_26/strided_slice_6StridedSlice'layer_normalization_26/Shape_1:output:05layer_normalization_26/strided_slice_6/stack:output:07layer_normalization_26/strided_slice_6/stack_1:output:07layer_normalization_26/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_26/mul_7Mul layer_normalization_26/mul_6:z:0/layer_normalization_26/strided_slice_6:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&layer_normalization_26/strided_slice_7StridedSlice'layer_normalization_26/Shape_1:output:05layer_normalization_26/strided_slice_7/stack:output:07layer_normalization_26/strided_slice_7/stack_1:output:07layer_normalization_26/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_26/mul_8/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_26/mul_8Mul'layer_normalization_26/mul_8/x:output:0/layer_normalization_26/strided_slice_7:output:0*
T0*
_output_shapes
: j
(layer_normalization_26/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :j
(layer_normalization_26/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
&layer_normalization_26/Reshape_2/shapePack1layer_normalization_26/Reshape_2/shape/0:output:0 layer_normalization_26/mul_7:z:0 layer_normalization_26/mul_8:z:01layer_normalization_26/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:¡
 layer_normalization_26/Reshape_2Reshape	add_1:z:0/layer_normalization_26/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
$layer_normalization_26/ones_1/packedPack layer_normalization_26/mul_7:z:0*
N*
T0*
_output_shapes
:h
#layer_normalization_26/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
layer_normalization_26/ones_1Fill-layer_normalization_26/ones_1/packed:output:0,layer_normalization_26/ones_1/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
%layer_normalization_26/zeros_1/packedPack layer_normalization_26/mul_7:z:0*
N*
T0*
_output_shapes
:i
$layer_normalization_26/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ³
layer_normalization_26/zeros_1Fill.layer_normalization_26/zeros_1/packed:output:0-layer_normalization_26/zeros_1/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
layer_normalization_26/Const_2Const*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_26/Const_3Const*
_output_shapes
: *
dtype0*
valueB ·
)layer_normalization_26/FusedBatchNormV3_1FusedBatchNormV3)layer_normalization_26/Reshape_2:output:0&layer_normalization_26/ones_1:output:0'layer_normalization_26/zeros_1:output:0'layer_normalization_26/Const_2:output:0'layer_normalization_26/Const_3:output:0*
T0*
U0*o
_output_shapes]
[:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:*
data_formatNCHW*
epsilon%o:½
 layer_normalization_26/Reshape_3Reshape-layer_normalization_26/FusedBatchNormV3_1:y:0'layer_normalization_26/Shape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,layer_normalization_26/Cast_2/ReadVariableOpReadVariableOp3layer_normalization_26_cast_readvariableop_resource*
_output_shapes
:@*
dtype0¾
layer_normalization_26/mul_9Mul)layer_normalization_26/Reshape_3:output:04layer_normalization_26/Cast_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,layer_normalization_26/Cast_3/ReadVariableOpReadVariableOp5layer_normalization_26_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0·
layer_normalization_26/add_1AddV2 layer_normalization_26/mul_9:z:04layer_normalization_26/Cast_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
ReluRelu layer_normalization_26/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ü
NoOpNoOp4^attention_head_rank_four_25/StatefulPartitionedCall ^dense_89/BiasAdd/ReadVariableOp"^dense_89/Tensordot/ReadVariableOp+^layer_normalization_26/Cast/ReadVariableOp-^layer_normalization_26/Cast_1/ReadVariableOp-^layer_normalization_26/Cast_2/ReadVariableOp-^layer_normalization_26/Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ@: : : : : : : 2j
3attention_head_rank_four_25/StatefulPartitionedCall3attention_head_rank_four_25/StatefulPartitionedCall2B
dense_89/BiasAdd/ReadVariableOpdense_89/BiasAdd/ReadVariableOp2F
!dense_89/Tensordot/ReadVariableOp!dense_89/Tensordot/ReadVariableOp2X
*layer_normalization_26/Cast/ReadVariableOp*layer_normalization_26/Cast/ReadVariableOp2\
,layer_normalization_26/Cast_1/ReadVariableOp,layer_normalization_26/Cast_1/ReadVariableOp2\
,layer_normalization_26/Cast_2/ReadVariableOp,layer_normalization_26/Cast_2/ReadVariableOp2\
,layer_normalization_26/Cast_3/ReadVariableOp,layer_normalization_26/Cast_3/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
þ	
ð
$__inference_signature_wrapper_273820
input_1
unknown:	'@
	unknown_0:@@
	unknown_1:@@
	unknown_2:@@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:	
	unknown_8:
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_273480o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1


__inference_call_274013

inputs4
"attention_head_rank_four_25_273882:@@4
"attention_head_rank_four_25_273884:@@4
"attention_head_rank_four_25_273886:@@A
3layer_normalization_26_cast_readvariableop_resource:@C
5layer_normalization_26_cast_1_readvariableop_resource:@<
*dense_89_tensordot_readvariableop_resource:@@6
(dense_89_biasadd_readvariableop_resource:@
identity¢3attention_head_rank_four_25/StatefulPartitionedCall¢dense_89/BiasAdd/ReadVariableOp¢!dense_89/Tensordot/ReadVariableOp¢*layer_normalization_26/Cast/ReadVariableOp¢,layer_normalization_26/Cast_1/ReadVariableOp¢,layer_normalization_26/Cast_2/ReadVariableOp¢,layer_normalization_26/Cast_3/ReadVariableOp¿
3attention_head_rank_four_25/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputs"attention_head_rank_four_25_273882"attention_head_rank_four_25_273884"attention_head_rank_four_25_273886*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_273321
addAddV2inputs<attention_head_rank_four_25/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
layer_normalization_26/ShapeShapeadd:z:0*
T0*
_output_shapes
:t
*layer_normalization_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,layer_normalization_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,layer_normalization_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$layer_normalization_26/strided_sliceStridedSlice%layer_normalization_26/Shape:output:03layer_normalization_26/strided_slice/stack:output:05layer_normalization_26/strided_slice/stack_1:output:05layer_normalization_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
layer_normalization_26/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_26/mulMul%layer_normalization_26/mul/x:output:0-layer_normalization_26/strided_slice:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
&layer_normalization_26/strided_slice_1StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_1/stack:output:07layer_normalization_26/strided_slice_1/stack_1:output:07layer_normalization_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_26/mul_1Mullayer_normalization_26/mul:z:0/layer_normalization_26/strided_slice_1:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
&layer_normalization_26/strided_slice_2StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_2/stack:output:07layer_normalization_26/strided_slice_2/stack_1:output:07layer_normalization_26/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_26/mul_2Mul layer_normalization_26/mul_1:z:0/layer_normalization_26/strided_slice_2:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
&layer_normalization_26/strided_slice_3StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_3/stack:output:07layer_normalization_26/strided_slice_3/stack_1:output:07layer_normalization_26/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_26/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_26/mul_3Mul'layer_normalization_26/mul_3/x:output:0/layer_normalization_26/strided_slice_3:output:0*
T0*
_output_shapes
: h
&layer_normalization_26/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&layer_normalization_26/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
$layer_normalization_26/Reshape/shapePack/layer_normalization_26/Reshape/shape/0:output:0 layer_normalization_26/mul_2:z:0 layer_normalization_26/mul_3:z:0/layer_normalization_26/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization_26/ReshapeReshapeadd:z:0-layer_normalization_26/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
"layer_normalization_26/ones/packedPack layer_normalization_26/mul_2:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_26/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ª
layer_normalization_26/onesFill+layer_normalization_26/ones/packed:output:0*layer_normalization_26/ones/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
#layer_normalization_26/zeros/packedPack layer_normalization_26/mul_2:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
layer_normalization_26/zerosFill,layer_normalization_26/zeros/packed:output:0+layer_normalization_26/zeros/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
layer_normalization_26/ConstConst*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_26/Const_1Const*
_output_shapes
: *
dtype0*
valueB ­
'layer_normalization_26/FusedBatchNormV3FusedBatchNormV3'layer_normalization_26/Reshape:output:0$layer_normalization_26/ones:output:0%layer_normalization_26/zeros:output:0%layer_normalization_26/Const:output:0'layer_normalization_26/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:*
data_formatNCHW*
epsilon%o:¹
 layer_normalization_26/Reshape_1Reshape+layer_normalization_26/FusedBatchNormV3:y:0%layer_normalization_26/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*layer_normalization_26/Cast/ReadVariableOpReadVariableOp3layer_normalization_26_cast_readvariableop_resource*
_output_shapes
:@*
dtype0¼
layer_normalization_26/mul_4Mul)layer_normalization_26/Reshape_1:output:02layer_normalization_26/Cast/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,layer_normalization_26/Cast_1/ReadVariableOpReadVariableOp5layer_normalization_26_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0µ
layer_normalization_26/addAddV2 layer_normalization_26/mul_4:z:04layer_normalization_26/Cast_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!dense_89/Tensordot/ReadVariableOpReadVariableOp*dense_89_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0a
dense_89/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:l
dense_89/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          f
dense_89/Tensordot/ShapeShapelayer_normalization_26/add:z:0*
T0*
_output_shapes
:b
 dense_89/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_89/Tensordot/GatherV2GatherV2!dense_89/Tensordot/Shape:output:0 dense_89/Tensordot/free:output:0)dense_89/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_89/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_89/Tensordot/GatherV2_1GatherV2!dense_89/Tensordot/Shape:output:0 dense_89/Tensordot/axes:output:0+dense_89/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_89/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_89/Tensordot/ProdProd$dense_89/Tensordot/GatherV2:output:0!dense_89/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_89/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_89/Tensordot/Prod_1Prod&dense_89/Tensordot/GatherV2_1:output:0#dense_89/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_89/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_89/Tensordot/concatConcatV2 dense_89/Tensordot/free:output:0 dense_89/Tensordot/axes:output:0'dense_89/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_89/Tensordot/stackPack dense_89/Tensordot/Prod:output:0"dense_89/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:§
dense_89/Tensordot/transpose	Transposelayer_normalization_26/add:z:0"dense_89/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
dense_89/Tensordot/ReshapeReshape dense_89/Tensordot/transpose:y:0!dense_89/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_89/Tensordot/MatMulMatMul#dense_89/Tensordot/Reshape:output:0)dense_89/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
dense_89/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@b
 dense_89/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_89/Tensordot/concat_1ConcatV2$dense_89/Tensordot/GatherV2:output:0#dense_89/Tensordot/Const_2:output:0)dense_89/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¢
dense_89/TensordotReshape#dense_89/Tensordot/MatMul:product:0$dense_89/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_89/BiasAdd/ReadVariableOpReadVariableOp(dense_89_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_89/BiasAddBiasAdddense_89/Tensordot:output:0'dense_89/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
add_1AddV2layer_normalization_26/add:z:0dense_89/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
layer_normalization_26/Shape_1Shape	add_1:z:0*
T0*
_output_shapes
:v
,layer_normalization_26/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.layer_normalization_26/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&layer_normalization_26/strided_slice_4StridedSlice'layer_normalization_26/Shape_1:output:05layer_normalization_26/strided_slice_4/stack:output:07layer_normalization_26/strided_slice_4/stack_1:output:07layer_normalization_26/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_26/mul_5/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_26/mul_5Mul'layer_normalization_26/mul_5/x:output:0/layer_normalization_26/strided_slice_4:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&layer_normalization_26/strided_slice_5StridedSlice'layer_normalization_26/Shape_1:output:05layer_normalization_26/strided_slice_5/stack:output:07layer_normalization_26/strided_slice_5/stack_1:output:07layer_normalization_26/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_26/mul_6Mul layer_normalization_26/mul_5:z:0/layer_normalization_26/strided_slice_5:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&layer_normalization_26/strided_slice_6StridedSlice'layer_normalization_26/Shape_1:output:05layer_normalization_26/strided_slice_6/stack:output:07layer_normalization_26/strided_slice_6/stack_1:output:07layer_normalization_26/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
layer_normalization_26/mul_7Mul layer_normalization_26/mul_6:z:0/layer_normalization_26/strided_slice_6:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&layer_normalization_26/strided_slice_7StridedSlice'layer_normalization_26/Shape_1:output:05layer_normalization_26/strided_slice_7/stack:output:07layer_normalization_26/strided_slice_7/stack_1:output:07layer_normalization_26/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_26/mul_8/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_26/mul_8Mul'layer_normalization_26/mul_8/x:output:0/layer_normalization_26/strided_slice_7:output:0*
T0*
_output_shapes
: j
(layer_normalization_26/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :j
(layer_normalization_26/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
&layer_normalization_26/Reshape_2/shapePack1layer_normalization_26/Reshape_2/shape/0:output:0 layer_normalization_26/mul_7:z:0 layer_normalization_26/mul_8:z:01layer_normalization_26/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:¡
 layer_normalization_26/Reshape_2Reshape	add_1:z:0/layer_normalization_26/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
$layer_normalization_26/ones_1/packedPack layer_normalization_26/mul_7:z:0*
N*
T0*
_output_shapes
:h
#layer_normalization_26/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
layer_normalization_26/ones_1Fill-layer_normalization_26/ones_1/packed:output:0,layer_normalization_26/ones_1/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
%layer_normalization_26/zeros_1/packedPack layer_normalization_26/mul_7:z:0*
N*
T0*
_output_shapes
:i
$layer_normalization_26/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ³
layer_normalization_26/zeros_1Fill.layer_normalization_26/zeros_1/packed:output:0-layer_normalization_26/zeros_1/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
layer_normalization_26/Const_2Const*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_26/Const_3Const*
_output_shapes
: *
dtype0*
valueB ·
)layer_normalization_26/FusedBatchNormV3_1FusedBatchNormV3)layer_normalization_26/Reshape_2:output:0&layer_normalization_26/ones_1:output:0'layer_normalization_26/zeros_1:output:0'layer_normalization_26/Const_2:output:0'layer_normalization_26/Const_3:output:0*
T0*
U0*o
_output_shapes]
[:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:*
data_formatNCHW*
epsilon%o:½
 layer_normalization_26/Reshape_3Reshape-layer_normalization_26/FusedBatchNormV3_1:y:0'layer_normalization_26/Shape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,layer_normalization_26/Cast_2/ReadVariableOpReadVariableOp3layer_normalization_26_cast_readvariableop_resource*
_output_shapes
:@*
dtype0¾
layer_normalization_26/mul_9Mul)layer_normalization_26/Reshape_3:output:04layer_normalization_26/Cast_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,layer_normalization_26/Cast_3/ReadVariableOpReadVariableOp5layer_normalization_26_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0·
layer_normalization_26/add_1AddV2 layer_normalization_26/mul_9:z:04layer_normalization_26/Cast_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
ReluRelu layer_normalization_26/add_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ü
NoOpNoOp4^attention_head_rank_four_25/StatefulPartitionedCall ^dense_89/BiasAdd/ReadVariableOp"^dense_89/Tensordot/ReadVariableOp+^layer_normalization_26/Cast/ReadVariableOp-^layer_normalization_26/Cast_1/ReadVariableOp-^layer_normalization_26/Cast_2/ReadVariableOp-^layer_normalization_26/Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ@: : : : : : : 2j
3attention_head_rank_four_25/StatefulPartitionedCall3attention_head_rank_four_25/StatefulPartitionedCall2B
dense_89/BiasAdd/ReadVariableOpdense_89/BiasAdd/ReadVariableOp2F
!dense_89/Tensordot/ReadVariableOp!dense_89/Tensordot/ReadVariableOp2X
*layer_normalization_26/Cast/ReadVariableOp*layer_normalization_26/Cast/ReadVariableOp2\
,layer_normalization_26/Cast_1/ReadVariableOp,layer_normalization_26/Cast_1/ReadVariableOp2\
,layer_normalization_26/Cast_2/ReadVariableOp,layer_normalization_26/Cast_2/ReadVariableOp2\
,layer_normalization_26/Cast_3/ReadVariableOp,layer_normalization_26/Cast_3/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Q
þ
__inference__traced_save_274378
file_prefixL
Hsavev2_transformer_encoder_1_embedding_38_embeddings_read_readvariableop.
*savev2_dense_89_kernel_read_readvariableop,
(savev2_dense_89_bias_read_readvariableop)
%savev2_variable_2_read_readvariableop)
%savev2_variable_1_read_readvariableop'
#savev2_variable_read_readvariableop;
7savev2_layer_normalization_26_gamma_read_readvariableop:
6savev2_layer_normalization_26_beta_read_readvariableopD
@savev2_transformer_encoder_1_dense_90_kernel_read_readvariableopB
>savev2_transformer_encoder_1_dense_90_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopS
Osavev2_adam_transformer_encoder_1_embedding_38_embeddings_m_read_readvariableop5
1savev2_adam_dense_89_kernel_m_read_readvariableop3
/savev2_adam_dense_89_bias_m_read_readvariableop0
,savev2_adam_variable_m_2_read_readvariableop0
,savev2_adam_variable_m_1_read_readvariableop.
*savev2_adam_variable_m_read_readvariableopB
>savev2_adam_layer_normalization_26_gamma_m_read_readvariableopA
=savev2_adam_layer_normalization_26_beta_m_read_readvariableopK
Gsavev2_adam_transformer_encoder_1_dense_90_kernel_m_read_readvariableopI
Esavev2_adam_transformer_encoder_1_dense_90_bias_m_read_readvariableopS
Osavev2_adam_transformer_encoder_1_embedding_38_embeddings_v_read_readvariableop5
1savev2_adam_dense_89_kernel_v_read_readvariableop3
/savev2_adam_dense_89_bias_v_read_readvariableop0
,savev2_adam_variable_v_2_read_readvariableop0
,savev2_adam_variable_v_1_read_readvariableop.
*savev2_adam_variable_v_read_readvariableopB
>savev2_adam_layer_normalization_26_gamma_v_read_readvariableopA
=savev2_adam_layer_normalization_26_beta_v_read_readvariableopK
Gsavev2_adam_transformer_encoder_1_dense_90_kernel_v_read_readvariableopI
Esavev2_adam_transformer_encoder_1_dense_90_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: §
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*Ð
valueÆBÃ(B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH½
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Í
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Hsavev2_transformer_encoder_1_embedding_38_embeddings_read_readvariableop*savev2_dense_89_kernel_read_readvariableop(savev2_dense_89_bias_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_1_read_readvariableop#savev2_variable_read_readvariableop7savev2_layer_normalization_26_gamma_read_readvariableop6savev2_layer_normalization_26_beta_read_readvariableop@savev2_transformer_encoder_1_dense_90_kernel_read_readvariableop>savev2_transformer_encoder_1_dense_90_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopOsavev2_adam_transformer_encoder_1_embedding_38_embeddings_m_read_readvariableop1savev2_adam_dense_89_kernel_m_read_readvariableop/savev2_adam_dense_89_bias_m_read_readvariableop,savev2_adam_variable_m_2_read_readvariableop,savev2_adam_variable_m_1_read_readvariableop*savev2_adam_variable_m_read_readvariableop>savev2_adam_layer_normalization_26_gamma_m_read_readvariableop=savev2_adam_layer_normalization_26_beta_m_read_readvariableopGsavev2_adam_transformer_encoder_1_dense_90_kernel_m_read_readvariableopEsavev2_adam_transformer_encoder_1_dense_90_bias_m_read_readvariableopOsavev2_adam_transformer_encoder_1_embedding_38_embeddings_v_read_readvariableop1savev2_adam_dense_89_kernel_v_read_readvariableop/savev2_adam_dense_89_bias_v_read_readvariableop,savev2_adam_variable_v_2_read_readvariableop,savev2_adam_variable_v_1_read_readvariableop*savev2_adam_variable_v_read_readvariableop>savev2_adam_layer_normalization_26_gamma_v_read_readvariableop=savev2_adam_layer_normalization_26_beta_v_read_readvariableopGsavev2_adam_transformer_encoder_1_dense_90_kernel_v_read_readvariableopEsavev2_adam_transformer_encoder_1_dense_90_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*­
_input_shapes
: :	'@:@@:@:@@:@@:@@:@:@:	:: : : : : : : : : :	'@:@@:@:@@:@@:@@:@:@:	::	'@:@@:@:@@:@@:@@:@:@:	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	'@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@:$ 

_output_shapes

:@@:$ 

_output_shapes

:@@: 

_output_shapes
:@: 

_output_shapes
:@:%	!

_output_shapes
:	: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	'@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@:$ 

_output_shapes

:@@:$ 

_output_shapes

:@@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	'@:$ 

_output_shapes

:@@:  

_output_shapes
:@:$! 

_output_shapes

:@@:$" 

_output_shapes

:@@:$# 

_output_shapes

:@@: $

_output_shapes
:@: %

_output_shapes
:@:%&!

_output_shapes
:	: '

_output_shapes
::(

_output_shapes
: 
É


6__inference_transformer_encoder_1_layer_call_fn_273845

transcript
unknown:	'@
	unknown_0:@@
	unknown_1:@@
	unknown_2:@@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:	
	unknown_8:
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCall
transcriptunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_transformer_encoder_1_layer_call_and_return_conditional_losses_273671o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
transcript
Ø
å
Q__inference_transformer_encoder_1_layer_call_and_return_conditional_losses_273671

transcript&
embedding_38_273497:	'@7
%transformer_block_rank_four_25_273635:@@7
%transformer_block_rank_four_25_273637:@@7
%transformer_block_rank_four_25_273639:@@3
%transformer_block_rank_four_25_273641:@3
%transformer_block_rank_four_25_273643:@7
%transformer_block_rank_four_25_273645:@@3
%transformer_block_rank_four_25_273647:@"
dense_90_273665:	
dense_90_273667:
identity¢ dense_90/StatefulPartitionedCall¢$embedding_38/StatefulPartitionedCall¢6transformer_block_rank_four_25/StatefulPartitionedCallø
$embedding_38/StatefulPartitionedCallStatefulPartitionedCall
transcriptembedding_38_273497*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_embedding_38_layer_call_and_return_conditional_losses_273496Ç
6transformer_block_rank_four_25/StatefulPartitionedCallStatefulPartitionedCall-embedding_38/StatefulPartitionedCall:output:0%transformer_block_rank_four_25_273635%transformer_block_rank_four_25_273637%transformer_block_rank_four_25_273639%transformer_block_rank_four_25_273641%transformer_block_rank_four_25_273643%transformer_block_rank_four_25_273645%transformer_block_rank_four_25_273647*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_transformer_block_rank_four_25_layer_call_and_return_conditional_losses_273634^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
flatten/ReshapeReshape?transformer_block_rank_four_25/StatefulPartitionedCall:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_90/StatefulPartitionedCallStatefulPartitionedCalldropout/Identity:output:0dense_90_273665dense_90_273667*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_90_layer_call_and_return_conditional_losses_273664x
IdentityIdentity)dense_90/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
NoOpNoOp!^dense_90/StatefulPartitionedCall%^embedding_38/StatefulPartitionedCall7^transformer_block_rank_four_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2D
 dense_90/StatefulPartitionedCall dense_90/StatefulPartitionedCall2L
$embedding_38/StatefulPartitionedCall$embedding_38/StatefulPartitionedCall2p
6transformer_block_rank_four_25/StatefulPartitionedCall6transformer_block_rank_four_25/StatefulPartitionedCall:W S
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
transcript
·I
þ
__inference_call_274218
inputs_for_keys
inputs_for_values
inputs_for_queries7
%einsum_einsum_readvariableop_resource:@@9
'einsum_1_einsum_readvariableop_resource:@@9
'einsum_2_einsum_readvariableop_resource:@@
identity¢einsum/Einsum/ReadVariableOp¢einsum_1/Einsum/ReadVariableOp¢einsum_2/Einsum/ReadVariableOp
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*
_output_shapes

:@@*
dtype0®
einsum/EinsumEinsuminputs_for_keys$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
equationixjk,kl->ixjl
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*
_output_shapes

:@@*
dtype0´
einsum_1/EinsumEinsuminputs_for_values&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
equationixjk,kl->ixjl
einsum_2/Einsum/ReadVariableOpReadVariableOp'einsum_2_einsum_readvariableop_resource*
_output_shapes

:@@*
dtype0µ
einsum_2/EinsumEinsuminputs_for_queries&einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
equationixjk,kl->ixjl
#attention_matrix_rank_four_25/ConstConst*"
_output_shapes
:*
dtype0*°
value¦B£"      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ          ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ              ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                          ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                              ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                          ÿ  ÿ  ÿ  ÿ  ÿ                                              ÿ  ÿ  ÿ  ÿ                                                  ÿ  ÿ  ÿ                                                      ÿ  ÿ                                                          ÿ                                                                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ          ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ              ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                          ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                              ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                          ÿ  ÿ  ÿ  ÿ  ÿ                                              ÿ  ÿ  ÿ  ÿ                                                  ÿ  ÿ  ÿ                                                      ÿ  ÿ                                                          ÿ                                                                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ          ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ              ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                          ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                              ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                          ÿ  ÿ  ÿ  ÿ  ÿ                                              ÿ  ÿ  ÿ  ÿ                                                  ÿ  ÿ  ÿ                                                      ÿ  ÿ                                                          ÿ                                                                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ          ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ              ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                          ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                              ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                      ÿ  ÿ  ÿ  ÿ  ÿ  ÿ                                          ÿ  ÿ  ÿ  ÿ  ÿ                                              ÿ  ÿ  ÿ  ÿ                                                  ÿ  ÿ  ÿ                                                      ÿ  ÿ                                                          ÿ                                                            
+attention_matrix_rank_four_25/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         Å
%attention_matrix_rank_four_25/ReshapeReshape,attention_matrix_rank_four_25/Const:output:04attention_matrix_rank_four_25/Reshape/shape:output:0*
T0*&
_output_shapes
:i
#attention_matrix_rank_four_25/ShapeShapeeinsum/Einsum:output:0*
T0*
_output_shapes
:{
1attention_matrix_rank_four_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3attention_matrix_rank_four_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3attention_matrix_rank_four_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+attention_matrix_rank_four_25/strided_sliceStridedSlice,attention_matrix_rank_four_25/Shape:output:0:attention_matrix_rank_four_25/strided_slice/stack:output:0<attention_matrix_rank_four_25/strided_slice/stack_1:output:0<attention_matrix_rank_four_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.attention_matrix_rank_four_25/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :p
.attention_matrix_rank_four_25/Tile/multiples/2Const*
_output_shapes
: *
dtype0*
value	B :p
.attention_matrix_rank_four_25/Tile/multiples/3Const*
_output_shapes
: *
dtype0*
value	B :Ã
,attention_matrix_rank_four_25/Tile/multiplesPack4attention_matrix_rank_four_25/strided_slice:output:07attention_matrix_rank_four_25/Tile/multiples/1:output:07attention_matrix_rank_four_25/Tile/multiples/2:output:07attention_matrix_rank_four_25/Tile/multiples/3:output:0*
N*
T0*
_output_shapes
:Ë
"attention_matrix_rank_four_25/TileTile.attention_matrix_rank_four_25/Reshape:output:05attention_matrix_rank_four_25/Tile/multiples:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,attention_matrix_rank_four_25/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ½
'attention_matrix_rank_four_25/transpose	Transposeeinsum/Einsum:output:05attention_matrix_rank_four_25/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
$attention_matrix_rank_four_25/MatMulBatchMatMulV2einsum_2/Einsum:output:0+attention_matrix_rank_four_25/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
'attention_matrix_rank_four_25/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *öÞw@Ë
%attention_matrix_rank_four_25/truedivRealDiv-attention_matrix_rank_four_25/MatMul:output:00attention_matrix_rank_four_25/truediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
!attention_matrix_rank_four_25/AddAddV2)attention_matrix_rank_four_25/truediv:z:0+attention_matrix_rank_four_25/Tile:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%attention_matrix_rank_four_25/SoftmaxSoftmax%attention_matrix_rank_four_25/Add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
MatMulBatchMatMulV2/attention_matrix_rank_four_25/Softmax:softmax:0einsum_1/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
IdentityIdentityMatMul:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
NoOpNoOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_2/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : 2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_2/Einsum/ReadVariableOpeinsum_2/Einsum/ReadVariableOp:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)
_user_specified_nameinputs_for_keys:b^
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+
_user_specified_nameinputs_for_values:c_
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,
_user_specified_nameinputs_for_queries"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¯
serving_default
?
input_14
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ß

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
embedding_layer
	encoder


classifier
	optimizer

signatures"
_tf_keras_model
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ð
trace_0
trace_12
6__inference_transformer_encoder_1_layer_call_fn_273694
6__inference_transformer_encoder_1_layer_call_fn_273845¦
²
FullArgSpec!
args
jself
j
transcript
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Ï
Q__inference_transformer_encoder_1_layer_call_and_return_conditional_losses_273879
Q__inference_transformer_encoder_1_layer_call_and_return_conditional_losses_273787¦
²
FullArgSpec!
args
jself
j
transcript
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
ÌBÉ
!__inference__wrapped_model_273480input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
Ý
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,ff_layer
-
self_atten
.
layer_norm
/call"
_tf_keras_layer
»
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer

6iter

7beta_1

8beta_2
	9decay
:learning_ratemmmmmmmmmmvvvvvvvvvv "
	optimizer
,
;serving_default"
signature_map
@:>	'@2-transformer_encoder_1/embedding_38/embeddings
!:@@2dense_89/kernel
:@2dense_89/bias
:@@2Variable
:@@2Variable
:@@2Variable
*:(@2layer_normalization_26/gamma
):'@2layer_normalization_26/beta
8:6	2%transformer_encoder_1/dense_90/kernel
1:/2#transformer_encoder_1/dense_90/bias
 "
trackable_list_wrapper
5
0
	1

2"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ïBì
6__inference_transformer_encoder_1_layer_call_fn_273694input_1"¦
²
FullArgSpec!
args
jself
j
transcript
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
òBï
6__inference_transformer_encoder_1_layer_call_fn_273845
transcript"¦
²
FullArgSpec!
args
jself
j
transcript
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_transformer_encoder_1_layer_call_and_return_conditional_losses_273879
transcript"¦
²
FullArgSpec!
args
jself
j
transcript
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_transformer_encoder_1_layer_call_and_return_conditional_losses_273787input_1"¦
²
FullArgSpec!
args
jself
j
transcript
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
ñ
Ctrace_02Ô
-__inference_embedding_38_layer_call_fn_274020¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zCtrace_0

Dtrace_02ï
H__inference_embedding_38_layer_call_and_return_conditional_losses_274029¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zDtrace_0
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object

Jtrace_02ú
?__inference_transformer_block_rank_four_25_layer_call_fn_274048¶
­²©
FullArgSpec1
args)&
jself
jinputs
jcontext_sequence
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zJtrace_0
²
Ktrace_02
Z__inference_transformer_block_rank_four_25_layer_call_and_return_conditional_losses_274182¶
­²©
FullArgSpec1
args)&
jself
jinputs
jcontext_sequence
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zKtrace_0
»
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ç
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses
k_weight
v_weight
q_weight
Xattn_mtx
Ycall"
_tf_keras_layer
Ä
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses
`axis
	gamma
beta"
_tf_keras_layer
ï
atrace_02Ò
__inference_call_274013¶
­²©
FullArgSpec1
args)&
jself
jinputs
jcontext_sequence
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zatrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
í
gtrace_02Ð
)__inference_dense_90_layer_call_fn_274227¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zgtrace_0

htrace_02ë
D__inference_dense_90_layer_call_and_return_conditional_losses_274238¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zhtrace_0
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ËBÈ
$__inference_signature_wrapper_273820input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
N
i	variables
j	keras_api
	ktotal
	lcount"
_tf_keras_metric
^
m	variables
n	keras_api
	ototal
	pcount
q
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
áBÞ
-__inference_embedding_38_layer_call_fn_274020inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
üBù
H__inference_embedding_38_layer_call_and_return_conditional_losses_274029inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
?__inference_transformer_block_rank_four_25_layer_call_fn_274048inputs"¶
­²©
FullArgSpec1
args)&
jself
jinputs
jcontext_sequence
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¢B
Z__inference_transformer_block_rank_four_25_layer_call_and_return_conditional_losses_274182inputs"¶
­²©
FullArgSpec1
args)&
jself
jinputs
jcontext_sequence
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
Ü2ÙÖ
Í²É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ü2ÙÖ
Í²É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
§
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer

trace_02ò
__inference_call_274218Ö
Í²É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
ßBÜ
__inference_call_274013inputs"¶
­²©
FullArgSpec1
args)&
jself
jinputs
jcontext_sequence
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_dense_90_layer_call_fn_274227inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_dense_90_layer_call_and_return_conditional_losses_274238inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
k0
l1"
trackable_list_wrapper
-
i	variables"
_generic_user_object
:  (2total
:  (2count
.
o0
p1"
trackable_list_wrapper
-
m	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
X0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¯B¬
__inference_call_274218inputs_for_keysinputs_for_valuesinputs_for_queries"Ö
Í²É
FullArgSpecQ
argsIF
jself
jinputs_for_keys
jinputs_for_values
jinputs_for_queries
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
E:C	'@24Adam/transformer_encoder_1/embedding_38/embeddings/m
&:$@@2Adam/dense_89/kernel/m
 :@2Adam/dense_89/bias/m
:@@2Adam/Variable/m
:@@2Adam/Variable/m
:@@2Adam/Variable/m
/:-@2#Adam/layer_normalization_26/gamma/m
.:,@2"Adam/layer_normalization_26/beta/m
=:;	2,Adam/transformer_encoder_1/dense_90/kernel/m
6:42*Adam/transformer_encoder_1/dense_90/bias/m
E:C	'@24Adam/transformer_encoder_1/embedding_38/embeddings/v
&:$@@2Adam/dense_89/kernel/v
 :@2Adam/dense_89/bias/v
:@@2Adam/Variable/v
:@@2Adam/Variable/v
:@@2Adam/Variable/v
/:-@2#Adam/layer_normalization_26/gamma/v
.:,@2"Adam/layer_normalization_26/beta/v
=:;	2,Adam/transformer_encoder_1/dense_90/kernel/v
6:42*Adam/transformer_encoder_1/dense_90/bias/v
!__inference__wrapped_model_273480w
4¢1
*¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
__inference_call_274013h;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
` 
ª " ÿÿÿÿÿÿÿÿÿ@ô
__inference_call_274218Ø®¢ª
¢¢
1.
inputs_for_keysÿÿÿÿÿÿÿÿÿ@
30
inputs_for_valuesÿÿÿÿÿÿÿÿÿ@
41
inputs_for_queriesÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@¥
D__inference_dense_90_layer_call_and_return_conditional_losses_274238]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_dense_90_layer_call_fn_274227P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ³
H__inference_embedding_38_layer_call_and_return_conditional_losses_274029g3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
-__inference_embedding_38_layer_call_fn_274020Z3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ@«
$__inference_signature_wrapper_273820
?¢<
¢ 
5ª2
0
input_1%"
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿÓ
Z__inference_transformer_block_rank_four_25_layer_call_and_return_conditional_losses_274182u;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
` 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 «
?__inference_transformer_block_rank_four_25_layer_call_fn_274048h;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
` 
ª " ÿÿÿÿÿÿÿÿÿ@¾
Q__inference_transformer_encoder_1_layer_call_and_return_conditional_losses_273787i
4¢1
*¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
Q__inference_transformer_encoder_1_layer_call_and_return_conditional_losses_273879l
7¢4
-¢*
(%

transcriptÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
6__inference_transformer_encoder_1_layer_call_fn_273694\
4¢1
*¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_transformer_encoder_1_layer_call_fn_273845_
7¢4
-¢*
(%

transcriptÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ