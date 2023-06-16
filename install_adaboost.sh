cd adaboost_bindings
maturin build
pip install --force-reinstall target/wheels/adaboost_bindings-0.1.0-cp310-cp310-manylinux_2_34_x86_64.whl
cd ..