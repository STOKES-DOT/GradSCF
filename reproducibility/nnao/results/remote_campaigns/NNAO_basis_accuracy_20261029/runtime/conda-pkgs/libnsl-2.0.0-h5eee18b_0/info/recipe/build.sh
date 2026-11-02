#! /bin/sh
# Get an updated config.sub and config.guess
cp $BUILD_PREFIX/share/gnuconfig/config.* .

./configure --prefix=${PREFIX} --disable-static
make
if [[ "${CONDA_BUILD_CROSS_COMPILATION:-}" != "1" ]]; then
make check
fi
make install
