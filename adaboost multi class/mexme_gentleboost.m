function mexme_gentleboost(options)
%
%
% Compile each mex-files of the yael gentleboost package
%
%  Usage
%  ------
%
%  mexme_gentleboost([options])
%
%  Inputs
%  -------
%
%  options            options strucure
%                     config_file   If user need a particular compiler config file (default config_file = []).
%                     ext           Extention to the compiled files (default ext = [])
%                     useOMP        Compile mex files with openMP support (0/1 for no/yes) (default useOMP = 0)
%                     userBLAS      Specific BLAS library's path (for linking with a specific version of MKL for example)
%
%
%
%  Example1 (with MSVC compiler or LCC compiler)
%  -------
%
%  mexme_gentleboost;
%
%  Example2 (with Intel Compiler )
%  -------
%
%  options.config_file = 'mexopts_intel10.bat';
%  options.ext         = 'mexw32';
%  options.useOMP      = 1;
%  options.userBLAS    = '"C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"';
%  mexme_gentleboost(options);
%
%  Example2 (with Intel Compiler & Windows 64)
%  -------
%
%  options.config_file = 'mexopts_intel11_64.bat';
%  options.useOMP      = 1;
%  options.userBLAS    = '"C:\Program Files (x86)\Intel\Compiler\11.1\065\mkl\em64t\lib\mkl_core.lib" "C:\Program Files (x86)\Intel\Compiler\11.1\065\mkl\em64t\lib\mkl_intel_lp64.lib" "C:\Program Files (x86)\Intel\Compiler\11.1\065\mkl\em64t\lib\mkl_intel_thread.lib"';
%  mexme_gentleboost(options);
warning off;
echo on
files1  = {'gentleboost_predict' , 'srng_model' , 'NN_predict'};
files2  = {'gentleboost_model'};
files3  = {'svmtrain.c svm.cpp svm_model_matlab.c' , 'svmpredict.c svm.cpp svm_model_matlab.c'};
if( (nargin < 1) || isempty(options) )
    options.config_file = [];
    options.ext         = [];
    options.useOMP      = 0;
    options.userBLAS    = [];
end
if(~any(strcmp(fieldnames(options) , 'config_file')))
    options.config_file = [];
end
if(~any(strcmp(fieldnames(options) , 'useOMP')))
    options.useOMP      = 0;
end
if(~exist(options.config_file , 'file'))
    options.config_file = [];
end
if(ispc)
    if (~exist(options.config_file , 'file'))
        optpath             = prefdir;
        mexoptfile          = [optpath filesep 'mexopts.bat'];
    else
        mexoptfile          = options.config_file;
    end
    
    res                     = getmexopts(mexoptfile);
    
    if(strcmp(res , 'lcc'))
        tempdir             =    fullfile(matlabroot , 'extern\lib\win32\lcc');
        if(options.useOMP==1)
            disp('OMP option can''t be used with LCC compiler, force to 0')
            options.useOMP      = 0;
        end
    elseif(strcmp(res , 'cl') || strcmp(res , 'icl'))
        tempdir             =    fullfile(matlabroot , sprintf('extern\\lib\\%s\\microsoft', computer('arch')));
    end
    if(isempty(options.userBLAS))
        if(exist(fullfile(tempdir , 'libmwblas.lib') , 'file') == 2)
            libblas         = ['"' , fullfile(tempdir , 'libmwblas.lib') , '"'];
        else % prior version of matlab %
            libblas         = ['"' , fullfile(tempdir , 'libmwlapack.lib') , '"'];
        end
    else
        libblas             = options.userBLAS;
    end
    
else
    libblas                 = '-lmwblas';
end
strOMP = [];
if(options.useOMP)
    if(ispc)
        if(strcmp(res , 'cl'))
            strOMP = ' COMPFLAGS="$COMPFLAGS /openmp" ';
        elseif(strcmp(res , 'icl'))
            strOMP = ' COMPFLAGS="$COMPFLAGS /Qopenmp" ';
        end
    else
        strOMP     = ' CFLAGS="\$CFLAGS -fopenmp -Wall" LDFLAGS="\$LDFLAGS -fopenmp" ';
    end
    
end
try
    
    for i = 1 : length(files1)
        
        str      = [];
        if(options.useOMP)
            str  = '-v -DOMP';
        end
        if(~isempty(options.config_file))
            str = [str, ' -f ' , options.config_file , ' '];
        end
        if(~isempty(options.ext))
            str = [str , '-output ' , files1{i} , '.' , options.ext , ' '];
        end
        str   = [str , files1{i} , '.c ' , libblas , strOMP];
        disp(['compiling ' files1{i}])
        eval(['mex ' str])
    end
    
    C = computer;
    
    for i = 1 : length(files2)
        
        str      = [];
        if(strcmp(C , 'PCWIN64') || strcmp(C , 'GLNXA64') || strcmp(C , 'MACI64') || strcmp(C , 'SOL64'))
            str  = '-DOS64 -largeArrayDims ';
        end
        if(options.useOMP)
            str  = [str , '-v -DOMP '];
        end
        if(~isempty(options.config_file))
            str  = [str, ' -f ' , options.config_file , ' '];
        end
        if(~isempty(options.ext))
            str  = [str , '-output ' , files2{i} , '.' , options.ext , ' '];
        end
        str      = [str , files2{i} , '.c ' , libblas , strOMP];
        disp(['compiling ' files2{i}])
        eval(['mex ' str])
    end
    
    
    for i = 1 : length(files3)
        str      = [];
        if(strcmp(C , 'PCWIN64') || strcmp(C , 'GLNXA64') || strcmp(C , 'MACI64') || strcmp(C , 'SOL64'))
            str  = '-largeArrayDims ';
        end
        
        temp               = files3{i};
        ind                = find(isspace(temp));
        [dummy , name]     = fileparts(temp(1:ind-1));
        
        if(~isempty(options.ext))
            str = [str , '-output ' , name , '.' , options.ext , ' '];
        end
        
        str                = [str , temp];
        
        disp(['compiling ' name])
        eval(['mex ' str])
    end
    
catch exception
    if(~isempty(exception))
        fprintf(['\n Error during compilation, be sure to:\n'...
            'i)  You have C compiler installed (prefered compiler are MSVC/Intel/GCC)\n'...
            'ii) You did "mex -setup" in matlab prompt before running mexme_gentleboost\n']);
    end
end
echo off
function res = getmexopts(mexoptfile)
% function res = getmexopts(Tag)
% Get the MCC or MEX configuration
% Author Bruno Luong <brunoluong@yahoo.com>
% Last update: 29-Jun-2009
Tag = 'COMPILER';
% Try to get MEX option first
fid=fopen(mexoptfile,'r');
if fid>0
    iscompilerline=@(S) (strcmp(S,['set ' Tag]));
    C=textscan(fid,'%s %s', 'delimiter', '=', 'whitespace', '');
    fclose(fid);
    cline=find(cellfun(iscompilerline,C{1}));
    if isempty(cline)
        error('getmexopt [Bruno]: cannot get Tag %s', Tag)
    end
    res=C{2}{cline};
    root=regexprep(matlabroot,'\\','\\\\');
    res = regexprep(res,'%MATLAB%',root);
else
    error('getmexopts [Bruno]: cannot open comopts.bat file')
end
