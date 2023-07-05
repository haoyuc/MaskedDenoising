python main_test_swinir.py \
        --model_path model_zoo/input_mask_80_90.pth  \
        --name input_mask_80_90/McM_poisson_20  \
        --opt model_zoo/input_mask_80_90.json \
        --folder_gt testset/McM/HR  \
        --folder_lq testset/McM/McM_poisson_20

python main_test_swinir.py \
        --model_path model_zoo/baseline.pth  \
        --name baseline/McM_poisson_20  \
        --opt model_zoo/baseline.json \
        --folder_gt testset/McM/HR  \
        --folder_lq testset/McM/McM_poisson_20        
