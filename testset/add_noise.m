file_path = './urban100/HR/'; % input HQ path
save_path = './urban100/urban100_speckle_0010/';  % output noisy path


img_path_list = dir(strcat(file_path,'*.png'));  
img_num = length(img_path_list);   
if img_num > 0   
    for j = 1:img_num   
        image_name = img_path_list(j).name;   
        I =  imread(strcat(file_path,image_name));
        fprintf('%d %s\n', j, strcat(file_path,image_name)); 

        path = strcat(save_path, image_name);
        
        % speckle
        speckle0024 = imnoise(I,'speckle', 0.024); 

        % salt & pepper
        % sp0002 = imnoise(I, 'salt & pepper', 0.002);   


        imwrite(speckle0024, path);

    end
end
