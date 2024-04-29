cd pylat
python -m numpy.f2py -c -m calccomf calcCOM.f90
python -m numpy.f2py -c -m calcdistances calcdistances.f90
python -m numpy.f2py -c -m ipcorr ipcorr.f90
#f2py -c -m pylat/elradial pylat/elradial.f90
#f2py -c -m pylat/siteradial pylat/siteradial.f90

ln -s ipcorr.*.so ipcorr.so
ln -s calcdistances.*.so calcdistances.so
ln -s calccomf.*.so calccomf.so

cd ..
