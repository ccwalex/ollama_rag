# ollama_rag
my implementation of a telegram bot powered by local RAG with ollama and langchain

some jupyter notebooks have some interrupted errors or changes that are not consistent


dependency:
+ ollama
+ unstructured
+ cuda (of course)
+ python / conda / jupyter

vector database:
+ qdrant (or your own choice)

telegram:
+ telebot

hardware I run with approximately 20 gb of pdf and some xml
+ intel 12th gen i7
+ 64 GB ram
+ 2 nvidia geforce 4080s
+ 60gb ssd swap


other dependency that may be used: (I didn't test):


_anaconda_depends         2024.06             py312_mkl_2  
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
abseil-cpp                20211102.0           hd4dd3e8_0  
absl-py                   2.1.0                    pypi_0    pypi
accelerate                1.0.1                    pypi_0    pypi
aiobotocore               2.12.3          py312h06a4308_0  
aiofiles                  24.1.0                   pypi_0    pypi
aiohttp                   3.9.5           py312h5eee18b_0  
aioitertools              0.7.1              pyhd3eb1b0_0  
aiosignal                 1.2.0              pyhd3eb1b0_0  
alabaster                 0.7.16          py312h06a4308_0  
alembic                   1.14.0                   pypi_0    pypi
altair                    5.0.1           py312h06a4308_0  
anaconda-catalogs         0.2.0           py312h06a4308_1  
annotated-types           0.6.0           py312h06a4308_0  
antlr4-python3-runtime    4.9.3                    pypi_0    pypi
anyio                     4.2.0           py312h06a4308_0  
aom                       3.6.0                h6a678d5_0  
appdirs                   1.4.4              pyhd3eb1b0_0  
apscheduler               3.6.3                    pypi_0    pypi
archspec                  0.2.3              pyhd3eb1b0_0  
argon2-cffi               21.3.0             pyhd3eb1b0_0  
argon2-cffi-bindings      21.2.0          py312h5eee18b_0  
arrow                     1.2.3           py312h06a4308_1  
arrow-cpp                 14.0.2               h374c478_1  
asgiref                   3.8.1                    pypi_0    pypi
astroid                   2.14.2          py312h06a4308_0  
astropy                   6.1.0           py312ha883a20_0  
astropy-iers-data         0.2024.6.3.0.31.14 py312h06a4308_0  
asttokens                 2.0.5              pyhd3eb1b0_0  
astunparse                1.6.3                    pypi_0    pypi
async-lru                 2.0.4           py312h06a4308_0  
asyncio                   3.4.3                    pypi_0    pypi
atomicwrites              1.4.0                      py_0  
attrs                     25.1.0                   pypi_0    pypi
automat                   20.2.0                     py_0  
autopep8                  2.0.4              pyhd3eb1b0_0  
aws-c-auth                0.6.19               h5eee18b_0  
aws-c-cal                 0.5.20               hdbd6064_0  
aws-c-common              0.8.5                h5eee18b_0  
aws-c-compression         0.2.16               h5eee18b_0  
aws-c-event-stream        0.2.15               h6a678d5_0  
aws-c-http                0.6.25               h5eee18b_0  
aws-c-io                  0.13.10              h5eee18b_0  
aws-c-mqtt                0.7.13               h5eee18b_0  
aws-c-s3                  0.1.51               hdbd6064_0  
aws-c-sdkutils            0.1.6                h5eee18b_0  
aws-checksums             0.1.13               h5eee18b_0  
aws-crt-cpp               0.18.16              h6a678d5_0  
aws-sdk-cpp               1.10.55              h721c034_0  
babel                     2.11.0          py312h06a4308_0  
backoff                   2.2.1                    pypi_0    pypi
bcrypt                    4.2.1                    pypi_0    pypi
beautifulsoup4            4.12.3          py312h06a4308_0  
binaryornot               0.4.4              pyhd3eb1b0_1  
black                     24.4.2          py312h06a4308_0  
blas                      1.0                         mkl  
bleach                    4.1.0              pyhd3eb1b0_0  
blinker                   1.6.2           py312h06a4308_0  
blosc                     1.21.3               h6a678d5_0  
bokeh                     3.4.1           py312he106c6f_0  
boltons                   23.0.0          py312h06a4308_0  
boost-cpp                 1.82.0               hdb19cb5_2  
botocore                  1.34.69         py312h06a4308_0  
bottleneck                1.3.7           py312ha883a20_0  
brotli                    1.0.9                h5eee18b_8  
brotli-bin                1.0.9                h5eee18b_8  
brotli-python             1.0.9           py312h6a678d5_8  
brunsli                   0.1                  h2531618_0  
build                     1.2.2.post1              pypi_0    pypi
bzip2                     1.0.8                h5eee18b_6  
c-ares                    1.19.1               h5eee18b_0  
c-blosc2                  2.12.0               h80c7b02_0  
ca-certificates           2024.12.31           h06a4308_0  
cachetools                5.3.3           py312h06a4308_0  
catboost                  1.2.7                    pypi_0    pypi
certifi                   2025.1.31       py312h06a4308_0  
cffi                      1.16.0          py312h5eee18b_1  
cfitsio                   3.470                h5893167_7  
chardet                   4.0.0           py312h06a4308_1003  
charls                    2.2.0                h2531618_0  
charset-normalizer        2.0.4              pyhd3eb1b0_0  
chatterbot                1.2.0                    pypi_0    pypi
chroma-hnswlib            0.7.6                    pypi_0    pypi
chromadb                  0.6.3                    pypi_0    pypi
click                     8.1.7           py312h06a4308_0  
cloudpickle               3.1.0                    pypi_0    pypi
colorama                  0.4.6           py312h06a4308_0  
colorcet                  3.1.0           py312h06a4308_0  
coloredlogs               15.0.1                   pypi_0    pypi
colorlog                  6.9.0                    pypi_0    pypi
comm                      0.2.1           py312h06a4308_0  
conda-content-trust       0.2.0           py312h06a4308_1  
conda-pack                0.7.1           py312h06a4308_0  
conda-package-handling    2.3.0           py312h06a4308_0  
conda-package-streaming   0.10.0          py312h06a4308_0  
conda-repo-cli            1.0.88          py312h06a4308_0  
constantly                23.10.4         py312h06a4308_0  
contourpy                 1.2.0           py312hdb19cb5_0  
cookiecutter              2.6.0           py312h06a4308_0  
cryptography              42.0.5          py312hdda0065_1  
cssselect                 1.2.0           py312h06a4308_0  
cucim-cu12                24.10.0                  pypi_0    pypi
cuda-python               12.6.0                   pypi_0    pypi
cudf-cu12                 24.10.1                  pypi_0    pypi
cugraph-cu12              24.10.0                  pypi_0    pypi
cuml-cu12                 24.10.0                  pypi_0    pypi
cuproj-cu12               24.10.0                  pypi_0    pypi
cupy-cuda12x              13.3.0                   pypi_0    pypi
curl                      8.7.1                hdbd6064_0  
cuspatial-cu12            24.10.0                  pypi_0    pypi
cuvs-cu12                 24.10.0                  pypi_0    pypi
cuxfilter-cu12            24.10.0                  pypi_0    pypi
cycler                    0.11.0             pyhd3eb1b0_0  
cyrus-sasl                2.1.28               h52b45da_1  
cytoolz                   0.12.2          py312h5eee18b_0  
daal                      2024.7.0                 pypi_0    pypi
daal4py                   2024.7.0                 pypi_0    pypi
dask                      2024.9.0                 pypi_0    pypi
dask-cuda                 24.10.0                  pypi_0    pypi
dask-cudf-cu12            24.10.1                  pypi_0    pypi
dask-expr                 1.1.14                   pypi_0    pypi
dataclasses-json          0.6.7                    pypi_0    pypi
datashader                0.16.2          py312h06a4308_0  
dav1d                     1.2.1                h5eee18b_0  
dbus                      1.13.18              hb2f20db_0  
debugpy                   1.6.7           py312h6a678d5_0  
decorator                 5.1.1              pyhd3eb1b0_0  
deepspeed                 0.15.2                   pypi_0    pypi
defusedxml                0.7.1              pyhd3eb1b0_0  
deprecated                1.2.18                   pypi_0    pypi
diff-match-patch          20200713           pyhd3eb1b0_0  
dill                      0.3.8           py312h06a4308_0  
distributed               2024.9.0                 pypi_0    pypi
distributed-ucxx-cu12     0.40.0                   pypi_0    pypi
distro                    1.9.0           py312h06a4308_0  
docstring-to-markdown     0.11            py312h06a4308_0  
docutils                  0.18.1          py312h06a4308_3  
dtaidistance              2.3.12                   pypi_0    pypi
durationpy                0.9                      pypi_0    pypi
effdet                    0.4.1                    pypi_0    pypi
emoji                     2.14.1                   pypi_0    pypi
entrypoints               0.4             py312h06a4308_0  
et_xmlfile                1.1.0           py312h06a4308_1  
eval-type-backport        0.2.2                    pypi_0    pypi
executing                 0.8.3              pyhd3eb1b0_0  
expat                     2.6.2                h6a678d5_0  
faiss-cpu                 1.10.0                   pypi_0    pypi
fastapi                   0.115.8                  pypi_0    pypi
fastrlock                 0.8.2                    pypi_0    pypi
filelock                  3.13.1          py312h06a4308_0  
filetype                  1.2.0                    pypi_0    pypi
flake8                    7.0.0           py312h06a4308_0  
flask                     3.0.3           py312h06a4308_0  
flatbuffers               24.3.25                  pypi_0    pypi
fmt                       9.1.0                hdb19cb5_1  
fontconfig                2.14.1               h4c34cd2_2  
fonttools                 4.51.0          py312h5eee18b_0  
fqdn                      1.5.1                    pypi_0    pypi
freetype                  2.12.1               h4a9f257_0  
freetype-py               2.5.1                    pypi_0    pypi
frozendict                2.4.2           py312h06a4308_0  
frozenlist                1.4.0           py312h5eee18b_0  
fsspec                    2024.3.1        py312h06a4308_0  
gast                      0.6.0                    pypi_0    pypi
gensim                    4.3.2           py312h526ad5a_0  
geopandas                 1.0.1                    pypi_0    pypi
gflags                    2.2.2                h6a678d5_1  
giflib                    5.2.1                h5eee18b_3  
gitdb                     4.0.7              pyhd3eb1b0_0  
gitpython                 3.1.37          py312h06a4308_0  
glib                      2.78.4               h6a678d5_0  
glib-tools                2.78.4               h6a678d5_0  
glog                      0.5.0                h6a678d5_1  
google-api-core           2.24.1                   pypi_0    pypi
google-auth               2.38.0                   pypi_0    pypi
google-cloud-vision       3.9.0                    pypi_0    pypi
google-pasta              0.2.0                    pypi_0    pypi
googleapis-common-protos  1.66.0                   pypi_0    pypi
greenlet                  3.0.1           py312h6a678d5_0  
grpc-cpp                  1.48.2               he1ff14a_1  
grpcio                    1.70.0                   pypi_0    pypi
grpcio-status             1.65.5                   pypi_0    pypi
grpcio-tools              1.70.0                   pypi_0    pypi
gst-plugins-base          1.14.1               h6a678d5_1  
gstreamer                 1.14.1               h5eee18b_1  
h11                       0.14.0          py312h06a4308_0  
h2                        4.2.0                    pypi_0    pypi
h5py                      3.11.0          py312h34c39bb_0  
hdbscan                   0.8.40                   pypi_0    pypi
hdf5                      1.12.1               h2b7332f_3  
heapdict                  1.0.1              pyhd3eb1b0_0  
hjson                     3.1.0                    pypi_0    pypi
holoviews                 1.19.0          py312h06a4308_0  
hpack                     4.1.0                    pypi_0    pypi
hsluv                     5.0.4                    pypi_0    pypi
html5lib                  1.1                      pypi_0    pypi
httpcore                  1.0.2           py312h06a4308_0  
httptools                 0.6.4                    pypi_0    pypi
httpx                     0.27.0          py312h06a4308_0  
httpx-sse                 0.4.0                    pypi_0    pypi
huggingface-hub           0.25.2                   pypi_0    pypi
humanfriendly             10.0                     pypi_0    pypi
hvplot                    0.10.0          py312h06a4308_0  
hyperframe                6.1.0                    pypi_0    pypi
hyperlink                 21.0.0             pyhd3eb1b0_0  
icu                       73.1                 h6a678d5_0  
idna                      3.7             py312h06a4308_0  
imagecodecs               2023.1.23       py312h81b8100_1  
imageio                   2.33.1          py312h06a4308_0  
imagesize                 1.4.1           py312h06a4308_0  
imbalanced-learn          0.12.3          py312h06a4308_1  
imblearn                  0.0                      pypi_0    pypi
importlib-metadata        7.0.1           py312h06a4308_0  
importlib-resources       6.5.2                    pypi_0    pypi
incremental               22.10.0            pyhd3eb1b0_0  
inflection                0.5.1           py312h06a4308_1  
iniconfig                 1.1.1              pyhd3eb1b0_0  
intake                    0.7.0           py312h06a4308_0  
intel-openmp              2023.1.0         hdb19cb5_46306  
intervaltree              3.1.0              pyhd3eb1b0_0  
ipykernel                 6.28.0          py312h06a4308_0  
ipympl                    0.9.4                    pypi_0    pypi
ipython                   8.25.0          py312h06a4308_0  
ipython_genutils          0.2.0              pyhd3eb1b0_1  
ipywidgets                7.8.1           py312h06a4308_0  
isoduration               20.11.0                  pypi_0    pypi
isort                     5.13.2          py312h06a4308_0  
itemadapter               0.3.0              pyhd3eb1b0_0  
itemloaders               1.1.0           py312h06a4308_0  
itsdangerous              2.2.0           py312h06a4308_0  
jaraco.classes            3.2.1              pyhd3eb1b0_0  
jedi                      0.18.1          py312h06a4308_1  
jeepney                   0.7.1              pyhd3eb1b0_0  
jellyfish                 1.0.1           py312hb02cf49_0  
jinja2                    3.1.4           py312h06a4308_0  
jmespath                  1.0.1           py312h06a4308_0  
joblib                    1.4.2           py312h06a4308_0  
jpeg                      9e                   h5eee18b_1  
jq                        1.6               h27cfd23_1000  
json5                     0.9.6              pyhd3eb1b0_0  
jsonpatch                 1.33            py312h06a4308_1  
jsonpath-python           1.0.6                    pypi_0    pypi
jsonpointer               2.1                pyhd3eb1b0_0  
jsonschema                4.19.2          py312h06a4308_0  
jsonschema-specifications 2023.7.1        py312h06a4308_0  
jupyter                   1.0.0           py312h06a4308_9  
jupyter-lsp               2.2.0           py312h06a4308_0  
jupyter-server-proxy      4.4.0                    pypi_0    pypi
jupyter_client            8.6.0           py312h06a4308_0  
jupyter_console           6.6.3           py312h06a4308_1  
jupyter_core              5.7.2           py312h06a4308_0  
jupyter_events            0.10.0          py312h06a4308_0  
jupyter_server            2.14.1          py312h06a4308_0  
jupyter_server_terminals  0.4.4           py312h06a4308_1  
jupyterlab                4.0.11          py312h06a4308_0  
jupyterlab-variableinspector 3.1.0           py312h06a4308_0  
jupyterlab_pygments       0.1.2                      py_0  
jupyterlab_server         2.25.1          py312h06a4308_0  
jupyterlab_widgets        1.0.0              pyhd3eb1b0_1  
jxrlib                    1.1                  h7b6447c_2  
kaggle                    1.6.17                   pypi_0    pypi
keras                     3.6.0                    pypi_0    pypi
keyring                   24.3.1          py312h06a4308_0  
kiwisolver                1.4.4           py312h6a678d5_0  
kmodes                    0.12.2                   pypi_0    pypi
krb5                      1.20.1               h143b758_1  
kubernetes                32.0.0                   pypi_0    pypi
langchain                 0.3.18                   pypi_0    pypi
langchain-chroma          0.2.2                    pypi_0    pypi
langchain-community       0.3.17                   pypi_0    pypi
langchain-core            0.3.34                   pypi_0    pypi
langchain-milvus          0.1.8                    pypi_0    pypi
langchain-ollama          0.2.3                    pypi_0    pypi
langchain-qdrant          0.2.0                    pypi_0    pypi
langchain-text-splitters  0.3.6                    pypi_0    pypi
langchain-unstructured    0.1.6                    pypi_0    pypi
langdetect                1.0.9                    pypi_0    pypi
langsmith                 0.3.8                    pypi_0    pypi
lazy-object-proxy         1.10.0          py312h5eee18b_0  
lazy_loader               0.4             py312h06a4308_0  
lcms2                     2.12                 h3be6417_0  
ld_impl_linux-64          2.38                 h1181459_1  
lerc                      3.0                  h295c915_0  
libaec                    1.0.4                he6710b0_1  
libarchive                3.6.2                h6ac8c49_3  
libavif                   0.11.1               h5eee18b_0  
libboost                  1.82.0               h109eef0_2  
libbrotlicommon           1.0.9                h5eee18b_8  
libbrotlidec              1.0.9                h5eee18b_8  
libbrotlienc              1.0.9                h5eee18b_8  
libclang                  18.1.1                   pypi_0    pypi
libclang13                14.0.6          default_he11475f_1  
libcudf-cu12              24.10.1                  pypi_0    pypi
libcups                   2.4.2                h2d74bed_1  
libcurl                   8.7.1                h251f7ec_0  
libcuspatial-cu12         24.10.0                  pypi_0    pypi
libdeflate                1.17                 h5eee18b_1  
libedit                   3.1.20230828         h5eee18b_0  
libev                     4.33                 h7f8727e_1  
libevent                  2.1.12               hdbd6064_1  
libffi                    3.4.4                h6a678d5_1  
libgcc-ng                 11.2.0               h1234567_1  
libgfortran-ng            11.2.0               h00389a5_1  
libgfortran5              11.2.0               h1234567_1  
libglib                   2.78.4               hdc74915_0  
libgomp                   11.2.0               h1234567_1  
libiconv                  1.16                 h5eee18b_3  
liblief                   0.12.3               h6a678d5_0  
libllvm14                 14.0.6               hdb19cb5_3  
libmamba                  1.5.8                hfe524e5_2  
libmambapy                1.5.8           py312h2dafd23_2  
libnghttp2                1.57.0               h2d74bed_0  
libpng                    1.6.39               h5eee18b_0  
libpq                     12.17                hdbd6064_0  
libprotobuf               3.20.3               he621ea3_0  
libsodium                 1.0.18               h7b6447c_0  
libsolv                   0.7.24               he621ea3_1  
libspatialindex           1.9.3                h2531618_0  
libssh2                   1.11.0               h251f7ec_0  
libstdcxx-ng              11.2.0               h1234567_1  
libthrift                 0.15.0               h1795dd8_2  
libtiff                   4.5.1                h6a678d5_0  
libucx-cu12               1.17.0                   pypi_0    pypi
libucxx-cu12              0.40.0                   pypi_0    pypi
libuuid                   1.41.5               h5eee18b_0  
libwebp-base              1.3.2                h5eee18b_0  
libxcb                    1.15                 h7f8727e_0  
libxkbcommon              1.0.1                h5eee18b_1  
libxml2                   2.10.4               hfdd30dd_2  
libxslt                   1.1.37               h5eee18b_1  
libzopfli                 1.0.3                he6710b0_0  
lightgbm                  4.5.0                    pypi_0    pypi
linkify-it-py             2.0.0           py312h06a4308_0  
llvmlite                  0.42.0          py312h6a678d5_0  
locket                    1.0.0           py312h06a4308_0  
lxml                      5.2.1           py312hdbbb534_0  
lz4                       4.3.2           py312h5eee18b_0  
lz4-c                     1.9.4                h6a678d5_1  
lzo                       2.10                 h7b6447c_2  
mako                      1.3.8                    pypi_0    pypi
markdown                  3.4.1           py312h06a4308_0  
markdown-it-py            2.2.0           py312h06a4308_1  
markupsafe                2.1.3           py312h5eee18b_0  
marshmallow               3.26.1                   pypi_0    pypi
mathparse                 0.1.2                    pypi_0    pypi
matplotlib                3.8.4           py312h06a4308_0  
matplotlib-base           3.8.4           py312h526ad5a_0  
matplotlib-inline         0.1.6           py312h06a4308_0  
mccabe                    0.7.0              pyhd3eb1b0_0  
mdit-py-plugins           0.3.0           py312h06a4308_0  
mdurl                     0.1.0           py312h06a4308_0  
menuinst                  2.1.1           py312h06a4308_0  
milvus-lite               2.4.11                   pypi_0    pypi
mistune                   2.0.4           py312h06a4308_0  
mkl                       2023.1.0         h213fc3f_46344  
mkl-service               2.4.0           py312h5eee18b_1  
mkl_fft                   1.3.8           py312h5eee18b_0  
mkl_random                1.2.4           py312hdb19cb5_0  
ml-dtypes                 0.4.1                    pypi_0    pypi
mmh3                      5.1.0                    pypi_0    pypi
monotonic                 1.6                      pypi_0    pypi
more-itertools            10.1.0          py312h06a4308_0  
mpmath                    1.3.0           py312h06a4308_0  
msgpack-python            1.0.3           py312hdb19cb5_0  
multidict                 6.0.4           py312h5eee18b_0  
multipledispatch          0.6.0           py312h06a4308_0  
multitasking              0.0.11                   pypi_0    pypi
mypy                      1.10.0          py312h5eee18b_0  
mypy_extensions           1.0.0           py312h06a4308_0  
mysql                     5.7.24               h721c034_2  
namex                     0.0.8                    pypi_0    pypi
nbclient                  0.8.0           py312h06a4308_0  
nbconvert                 7.10.0          py312h06a4308_0  
nbformat                  5.9.2           py312h06a4308_0  
ncurses                   6.4                  h6a678d5_0  
nest-asyncio              1.6.0           py312h06a4308_0  
networkx                  3.2.1           py312h06a4308_0  
ninja                     1.11.1.1                 pypi_0    pypi
nltk                      3.8.1           py312h06a4308_0  
notebook                  7.0.8           py312h06a4308_0  
notebook-shim             0.2.3           py312h06a4308_0  
nspr                      4.35                 h6a678d5_0  
nss                       3.89.1               h6a678d5_0  
numba                     0.59.1          py312h526ad5a_0  
numexpr                   2.8.7           py312hf827012_0  
numpy                     1.26.4          py312hc5e2394_0  
numpy-base                1.26.4          py312h0da6c21_0  
numpydoc                  1.7.0           py312h06a4308_0  
nvidia-cublas-cu12        12.4.2.65                pypi_0    pypi
nvidia-cuda-cupti-cu12    12.4.99                  pypi_0    pypi
nvidia-cuda-nvcc-cu12     12.3.107                 pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.4.99                  pypi_0    pypi
nvidia-cuda-runtime-cu12  12.4.99                  pypi_0    pypi
nvidia-cudnn-cu12         9.1.0.70                 pypi_0    pypi
nvidia-cufft-cu12         11.2.0.44                pypi_0    pypi
nvidia-curand-cu12        10.3.5.119               pypi_0    pypi
nvidia-cusolver-cu12      11.6.0.99                pypi_0    pypi
nvidia-cusparse-cu12      12.3.0.142               pypi_0    pypi
nvidia-ml-py              12.560.30                pypi_0    pypi
nvidia-nccl-cu12          2.20.5                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.4.99                  pypi_0    pypi
nvidia-nvtx-cu12          12.4.99                  pypi_0    pypi
nvtx                      0.2.10                   pypi_0    pypi
nx-cugraph-cu12           24.10.0                  pypi_0    pypi
oauthlib                  3.2.2                    pypi_0    pypi
olefile                   0.47                     pypi_0    pypi
ollama                    0.4.7                    pypi_0    pypi
omegaconf                 2.3.0                    pypi_0    pypi
oniguruma                 6.9.7.1              h27cfd23_0  
onnx                      1.17.0                   pypi_0    pypi
onnxruntime               1.19.2                   pypi_0    pypi
opencv-python             4.11.0.86                pypi_0    pypi
openjpeg                  2.4.0                h3ad879b_0  
openpyxl                  3.1.2           py312h5eee18b_0  
openssl                   3.0.15               h5eee18b_0  
opentelemetry-api         1.30.0                   pypi_0    pypi
opentelemetry-exporter-otlp-proto-common 1.30.0                   pypi_0    pypi
opentelemetry-exporter-otlp-proto-grpc 1.30.0                   pypi_0    pypi
opentelemetry-instrumentation 0.51b0                   pypi_0    pypi
opentelemetry-instrumentation-asgi 0.51b0                   pypi_0    pypi
opentelemetry-instrumentation-fastapi 0.51b0                   pypi_0    pypi
opentelemetry-proto       1.30.0                   pypi_0    pypi
opentelemetry-sdk         1.30.0                   pypi_0    pypi
opentelemetry-semantic-conventions 0.51b0                   pypi_0    pypi
opentelemetry-util-http   0.51b0                   pypi_0    pypi
opt-einsum                3.4.0                    pypi_0    pypi
optree                    0.13.0                   pypi_0    pypi
optuna                    4.1.0                    pypi_0    pypi
orc                       1.7.4                hb3bc3d3_1  
orjson                    3.10.15                  pypi_0    pypi
outcome                   1.3.0.post0              pypi_0    pypi
overrides                 7.4.0           py312h06a4308_0  
packaging                 23.2            py312h06a4308_0  
pandas                    2.2.2           py312h526ad5a_0  
pandas-datareader         0.10.0                   pypi_0    pypi
pandas-ta                 0.3.14b0                 pypi_0    pypi
pandocfilters             1.5.0              pyhd3eb1b0_0  
panel                     1.4.4           py312h06a4308_0  
param                     2.1.0           py312h06a4308_0  
parsel                    1.8.1           py312h06a4308_0  
parso                     0.8.3              pyhd3eb1b0_0  
partd                     1.4.1           py312h06a4308_0  
patch                     2.7.6             h7b6447c_1001  
patchelf                  0.17.2               h6a678d5_0  
pathspec                  0.10.3          py312h06a4308_0  
patsy                     0.5.6           py312h06a4308_0  
pcre2                     10.42                hebb0a14_1  
pdf2image                 1.17.0                   pypi_0    pypi
pdfminer-six              20240706                 pypi_0    pypi
peewee                    3.17.6                   pypi_0    pypi
pexpect                   4.8.0              pyhd3eb1b0_3  
pi-heif                   0.21.0                   pypi_0    pypi
pickleshare               0.7.5           pyhd3eb1b0_1003  
pikepdf                   9.5.2                    pypi_0    pypi
pillow                    10.3.0          py312h5eee18b_0  
pip                       24.0            py312h06a4308_0  
pkce                      1.0.3           py312h06a4308_0  
pkginfo                   1.10.0          py312h06a4308_0  
platformdirs              3.10.0          py312h06a4308_0  
plotly                    5.22.0          py312he106c6f_0  
pluggy                    1.0.0           py312h06a4308_1  
ply                       3.11            py312h06a4308_1  
portalocker               2.10.1                   pypi_0    pypi
posthog                   3.13.0                   pypi_0    pypi
prometheus_client         0.14.1          py312h06a4308_0  
prompt-toolkit            3.0.43          py312h06a4308_0  
prompt_toolkit            3.0.43               hd3eb1b0_0  
protego                   0.1.16                     py_0  
proto-plus                1.26.0                   pypi_0    pypi
protobuf                  5.29.3                   pypi_0    pypi
psutil                    5.9.0           py312h5eee18b_0  
ptyprocess                0.7.0              pyhd3eb1b0_2  
pure_eval                 0.2.2              pyhd3eb1b0_0  
py-cpuinfo                9.0.0           py312h06a4308_0  
py-lief                   0.12.3          py312h6a678d5_0  
pyarrow                   14.0.2          py312hb107042_0  
pyasn1                    0.4.8              pyhd3eb1b0_0  
pyasn1-modules            0.2.8                      py_0  
pybind11-abi              5                    hd3eb1b0_0  
pycocotools               2.0.8                    pypi_0    pypi
pycodestyle               2.11.1          py312h06a4308_0  
pycosat                   0.6.6           py312h5eee18b_1  
pycparser                 2.21               pyhd3eb1b0_0  
pyct                      0.5.0           py312h06a4308_0  
pycurl                    7.45.2          py312hdbd6064_1  
pydantic                  2.10.6                   pypi_0    pypi
pydantic-core             2.27.2                   pypi_0    pypi
pydantic-settings         2.7.1                    pypi_0    pypi
pydeck                    0.8.0           py312h06a4308_2  
pydispatcher              2.0.5           py312h06a4308_3  
pydocstyle                6.3.0           py312h06a4308_0  
pyerfa                    2.0.1.4         py312ha883a20_0  
pyflakes                  3.2.0           py312h06a4308_0  
pygments                  2.15.1          py312h06a4308_1  
pyjwt                     2.8.0           py312h06a4308_0  
pylibcudf-cu12            24.10.1                  pypi_0    pypi
pylibcugraph-cu12         24.10.0                  pypi_0    pypi
pylibraft-cu12            24.10.0                  pypi_0    pypi
pylint                    2.16.2          py312h06a4308_0  
pylint-venv               3.0.3           py312h06a4308_0  
pyls-spyder               0.4.0              pyhd3eb1b0_0  
pymilvus                  2.5.4                    pypi_0    pypi
pynndescent               0.5.13                   pypi_0    pypi
pynvjitlink-cu12          0.3.0                    pypi_0    pypi
pynvml                    11.4.1                   pypi_0    pypi
pyodbc                    5.0.1           py312h6a678d5_0  
pyogrio                   0.10.0                   pypi_0    pypi
pyopenssl                 24.0.0          py312h06a4308_0  
pypandoc                  1.15                     pypi_0    pypi
pyparsing                 3.0.9           py312h06a4308_0  
pypdf                     5.3.0                    pypi_0    pypi
pypdfium2                 4.30.1                   pypi_0    pypi
pypika                    0.48.9                   pypi_0    pypi
pyproj                    3.7.0                    pypi_0    pypi
pyproject-hooks           1.2.0                    pypi_0    pypi
pyqt                      5.15.10         py312h6a678d5_0  
pyqt5-sip                 12.13.0         py312h5eee18b_0  
pyqtwebengine             5.15.10         py312h6a678d5_0  
pysocks                   1.7.1           py312h06a4308_0  
pytables                  3.9.2           py312h387d6ec_0  
pytelegrambotapi          4.26.0                   pypi_0    pypi
pytest                    7.4.4           py312h06a4308_0  
python                    3.12.4               h5148396_1  
python-dateutil           2.9.0post0      py312h06a4308_2  
python-docx               1.1.2                    pypi_0    pypi
python-dotenv             1.0.1                    pypi_0    pypi
python-fastjsonschema     2.16.2          py312h06a4308_0  
python-graphviz           0.20.3                   pypi_0    pypi
python-iso639             2025.2.8                 pypi_0    pypi
python-json-logger        2.0.7           py312h06a4308_0  
python-libarchive-c       2.9                pyhd3eb1b0_1  
python-lmdb               1.4.1           py312h6a678d5_0  
python-lsp-black          2.0.0           py312h06a4308_0  
python-lsp-jsonrpc        1.1.2              pyhd3eb1b0_0  
python-lsp-server         1.10.0          py312h06a4308_0  
python-magic              0.4.27                   pypi_0    pypi
python-multipart          0.0.20                   pypi_0    pypi
python-oxmsg              0.0.2                    pypi_0    pypi
python-pptx               1.0.2                    pypi_0    pypi
python-slugify            5.0.2              pyhd3eb1b0_0  
python-snappy             0.6.1           py312h6a678d5_0  
python-telegram-bot       21.10                    pypi_0    pypi
python-tzdata             2023.3             pyhd3eb1b0_0  
pytoolconfig              1.2.6           py312h06a4308_0  
pytz                      2025.1                   pypi_0    pypi
pyviz_comms               3.0.2           py312h06a4308_0  
pywavelets                1.5.0           py312ha883a20_0  
pyxdg                     0.27               pyhd3eb1b0_0  
pyyaml                    6.0.1           py312h5eee18b_0  
pyzmq                     25.1.2          py312h6a678d5_0  
qdarkstyle                3.2.3              pyhd3eb1b0_0  
qdrant-client             1.13.2                   pypi_0    pypi
qstylizer                 0.2.2           py312h06a4308_0  
qt-main                   5.15.2              h53bd1ea_10  
qt-webengine              5.15.9               h9ab4d14_7  
qtawesome                 1.2.2           py312h06a4308_0  
qtconsole                 5.5.1           py312h06a4308_0  
qtpy                      2.4.1           py312h06a4308_0  
queuelib                  1.6.2           py312h06a4308_0  
raft-dask-cu12            24.10.0                  pypi_0    pypi
rapidfuzz                 3.12.1                   pypi_0    pypi
rapids-dask-dependency    24.10.0                  pypi_0    pypi
re2                       2022.04.01           h295c915_0  
readline                  8.2                  h5eee18b_0  
referencing               0.30.2          py312h06a4308_0  
regex                     2023.10.3       py312h5eee18b_0  
reproc                    14.2.4               h6a678d5_2  
reproc-cpp                14.2.4               h6a678d5_2  
requests                  2.32.3          py312h06a4308_1  
requests-file             1.5.1              pyhd3eb1b0_0  
requests-oauthlib         2.0.0                    pypi_0    pypi
requests-toolbelt         1.0.0           py312h06a4308_0  
rfc3339-validator         0.1.4           py312h06a4308_0  
rfc3986-validator         0.1.1           py312h06a4308_0  
rich                      13.3.5          py312h06a4308_1  
rmm-cu12                  24.10.0                  pypi_0    pypi
rope                      1.12.0          py312h06a4308_0  
rpds-py                   0.10.6          py312hb02cf49_0  
rsa                       4.9                      pypi_0    pypi
rtree                     1.0.1           py312h06a4308_0  
ruamel.yaml               0.17.21         py312h5eee18b_0  
ruamel_yaml               0.17.21         py312h5eee18b_0  
s2n                       1.3.27               hdbd6064_0  
s3fs                      2024.3.1        py312h06a4308_0  
safetensors               0.4.5                    pypi_0    pypi
scikit-image              0.23.2          py312h526ad5a_0  
scikit-learn              1.4.2           py312h526ad5a_1  
scikit-learn-intelex      2024.7.0                 pypi_0    pypi
scipy                     1.13.1          py312hc5e2394_0  
scrapy                    2.11.1          py312h06a4308_0  
seaborn                   0.13.2          py312h06a4308_0  
secretstorage             3.3.1           py312h06a4308_1  
selenium                  4.28.1                   pypi_0    pypi
semver                    3.0.2           py312h06a4308_0  
send2trash                1.8.2           py312h06a4308_0  
service_identity          18.1.0             pyhd3eb1b0_1  
setuptools                69.5.1          py312h06a4308_0  
shapely                   2.0.6                    pypi_0    pypi
shellingham               1.5.4                    pypi_0    pypi
simpervisor               1.0.0                    pypi_0    pypi
sip                       6.7.12          py312h6a678d5_0  
six                       1.16.0             pyhd3eb1b0_1  
smart_open                5.2.1           py312h06a4308_0  
smmap                     4.0.0              pyhd3eb1b0_0  
snappy                    1.1.10               h6a678d5_1  
sniffio                   1.3.0           py312h06a4308_0  
snowballstemmer           2.2.0              pyhd3eb1b0_0  
sortedcontainers          2.4.0              pyhd3eb1b0_0  
soupsieve                 2.5             py312h06a4308_0  
sphinx                    7.3.7           py312h5eee18b_0  
sphinxcontrib-applehelp   1.0.2              pyhd3eb1b0_0  
sphinxcontrib-devhelp     1.0.2              pyhd3eb1b0_0  
sphinxcontrib-htmlhelp    2.0.0              pyhd3eb1b0_0  
sphinxcontrib-jsmath      1.0.1              pyhd3eb1b0_0  
sphinxcontrib-qthelp      1.0.3              pyhd3eb1b0_0  
sphinxcontrib-serializinghtml 1.1.10          py312h06a4308_0  
spyder                    5.5.1           py312h06a4308_0  
spyder-kernels            2.5.0           py312h06a4308_0  
sqlalchemy                2.0.30          py312h5eee18b_0  
sqlite                    3.45.3               h5eee18b_0  
stack_data                0.2.0              pyhd3eb1b0_0  
starlette                 0.45.3                   pypi_0    pypi
statsmodels               0.14.2          py312ha883a20_0  
streamlit                 1.42.0                   pypi_0    pypi
sympy                     1.12            py312h06a4308_0  
tabulate                  0.9.0           py312h06a4308_0  
tbb                       2021.13.1                pypi_0    pypi
tblib                     1.7.0              pyhd3eb1b0_0  
tenacity                  9.0.0                    pypi_0    pypi
tensorboard               2.18.0                   pypi_0    pypi
tensorboard-data-server   0.7.2                    pypi_0    pypi
tensorflow                2.18.0                   pypi_0    pypi
tensorrt                  10.6.0.post1             pypi_0    pypi
tensorrt-cu12             10.6.0.post1             pypi_0    pypi
tensorrt-cu12-bindings    10.6.0.post1             pypi_0    pypi
tensorrt-cu12-libs        10.6.0.post1             pypi_0    pypi
termcolor                 2.5.0                    pypi_0    pypi
terminado                 0.17.1          py312h06a4308_0  
text-unidecode            1.3                pyhd3eb1b0_0  
textdistance              4.2.1              pyhd3eb1b0_0  
threadpoolctl             3.5.0              pyhc1e730c_0    conda-forge
three-merge               0.1.1              pyhd3eb1b0_0  
tifffile                  2023.4.12       py312h06a4308_0  
timm                      1.0.14                   pypi_0    pypi
tinycss2                  1.2.1           py312h06a4308_0  
tk                        8.6.14               h39e8969_0  
tldextract                3.2.0              pyhd3eb1b0_0  
tokenizers                0.20.1                   pypi_0    pypi
toml                      0.10.2             pyhd3eb1b0_0  
tomli                     2.0.1           py312h06a4308_1  
tomlkit                   0.11.1          py312h06a4308_0  
toolz                     0.12.0          py312h06a4308_0  
torch                     2.4.1+cu124              pypi_0    pypi
torchaudio                2.4.1+cu124              pypi_0    pypi
torchvision               0.19.1+cu124             pypi_0    pypi
tornado                   6.4.1           py312h5eee18b_0  
tqdm                      4.66.4          py312he106c6f_0  
traitlets                 5.14.3          py312h06a4308_0  
transformers              4.45.2                   pypi_0    pypi
treelite                  4.3.0                    pypi_0    pypi
trio                      0.28.0                   pypi_0    pypi
trio-websocket            0.11.1                   pypi_0    pypi
triton                    3.0.0                    pypi_0    pypi
truststore                0.8.0           py312h06a4308_0  
tslearn                   0.6.3                    pypi_0    pypi
twisted                   23.10.0         py312h06a4308_0  
typer                     0.15.1                   pypi_0    pypi
typing-extensions         4.12.2                   pypi_0    pypi
typing-inspect            0.9.0                    pypi_0    pypi
tzdata                    2024a                h04d1e81_0  
tzlocal                   2.1                      pypi_0    pypi
uc-micro-py               1.0.1           py312h06a4308_0  
ucx-py-cu12               0.40.0                   pypi_0    pypi
ucxx-cu12                 0.40.0                   pypi_0    pypi
ujson                     5.10.0          py312h6a678d5_0  
umap-learn                0.5.6                    pypi_0    pypi
unicodedata2              15.1.0          py312h5eee18b_0  
unidecode                 1.2.0              pyhd3eb1b0_0  
unixodbc                  2.3.11               h5eee18b_0  
unstructured              0.16.20                  pypi_0    pypi
unstructured-client       0.29.0                   pypi_0    pypi
unstructured-inference    0.8.7                    pypi_0    pypi
unstructured-pytesseract  0.3.13                   pypi_0    pypi
uri-template              1.3.0                    pypi_0    pypi
urllib3                   2.2.2           py312h06a4308_0  
useragent                 0.1.1                    pypi_0    pypi
utf8proc                  2.6.1                h5eee18b_1  
uvicorn                   0.34.0                   pypi_0    pypi
uvloop                    0.21.0                   pypi_0    pypi
vispy                     0.14.3                   pypi_0    pypi
w3lib                     1.21.0             pyhd3eb1b0_0  
watchdog                  4.0.1           py312h06a4308_0  
watchfiles                1.0.4                    pypi_0    pypi
wcwidth                   0.2.5              pyhd3eb1b0_0  
webcolors                 24.8.0                   pypi_0    pypi
webencodings              0.5.1           py312h06a4308_2  
websocket-client          1.8.0           py312h06a4308_0  
websockets                14.2                     pypi_0    pypi
werkzeug                  3.0.3           py312h06a4308_0  
wget                      3.2                      pypi_0    pypi
whatthepatch              1.0.2           py312h06a4308_0  
wheel                     0.43.0          py312h06a4308_0  
widgetsnbextension        3.6.6           py312h06a4308_0  
wrapt                     1.14.1          py312h5eee18b_0  
wsproto                   1.2.0                    pypi_0    pypi
wurlitzer                 3.0.2           py312h06a4308_0  
xarray                    2023.6.0        py312h06a4308_0  
xlrd                      2.0.1                    pypi_0    pypi
xlsxwriter                3.2.2                    pypi_0    pypi
xyzservices               2022.9.0        py312h06a4308_1  
xz                        5.4.6                h5eee18b_1  
yaml                      0.2.5                h7b6447c_0  
yaml-cpp                  0.8.0                h6a678d5_1  
yapf                      0.40.2          py312h06a4308_0  
yarl                      1.9.3           py312h5eee18b_0  
yfinance                  0.2.44                   pypi_0    pypi
zeromq                    4.3.5                h6a678d5_0  
zfp                       1.0.0                h6a678d5_0  
zict                      3.0.0           py312h06a4308_0  
zipp                      3.17.0          py312h06a4308_0  
zlib                      1.2.13               h5eee18b_1  
zlib-ng                   2.0.7                h5eee18b_0  
zope                      1.0             py312h06a4308_1  
zope.interface            5.4.0           py312h5eee18b_0  
zstandard                 0.23.0                   pypi_0    pypi
zstd                      1.5.5                hc292b87_2
