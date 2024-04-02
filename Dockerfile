# docker tf 2.4.0 works for CUDA 11
FROM tensorflow/tensorflow:2.4.0-gpu-jupyter

# install additional required packages other than TF and Jupyter
# NOTE: keras is included in TF
RUN pip install matplotlib==3.1.1 scikit-learn==0.21.3
