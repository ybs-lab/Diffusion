function tracking_mat2csv(folderpath)% Convert .mat particle tracking file to a .csv format similar to pytrack:
%folderpath is folder with mat files to read, save csv files there
    columns = {'x', 'y', 'size', 'mass', 'frame', 'particle'};

    fileList = dir(folderpath);
    fileList = fileList(3:end);
    for n = 1:length(fileList)
        curFile = fileList(n);
        fullPath = [curFile.folder,'\',curFile.name];
        [~,name,extension] = fileparts(fullPath);
        if strcmpi(extension,'.mat')
            output_fullPath  = [curFile.folder,'\',name,'.csv'];
            if ~isfile(output_fullPath)
                dataStruct=load(fullPath);
                fieldname = fieldnames(dataStruct);                
                data = dataStruct.(fieldname{1});
                T = array2table(data);
                T.Properties.VariableNames=columns;        
                writetable(T,output_fullPath);        
                sprintf('Created file %s',output_fullPath)
            else
                sprintf('File %s already exists!',output_fullPath)
            end
        end
    end

end