import hashlib
import requests
import os
from tqdm import tqdm

url_path_list = [
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.3/1k3d68.onnx', 'path': './models/1k3d68.onnx', 'expected_hash': 'df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.3/face_blendshapes_Nx146x2.onnx', 'path': './models/face_blendshapes_Nx146x2.onnx', 'expected_hash': '79065a18016da3b95f71247ff9ade3fe09b9124903a26a1af85af6d9e2a4faf3'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.3/simswap_512_unoff.onnx', 'path': './models/simswap_512_unoff.onnx', 'expected_hash': '08c6ca9c0a65eff119bea42686a4574337141de304b9d26e2f9d11e78d9e8e86'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.3/simswap_arcface_model.onnx', 'path': './models/simswap_arcface_model.onnx', 'expected_hash': '58949c864ab4a89012aaefc117f1ab8548c5f470bbc3889474bca13a412fc843'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.4/ghost_arcface_backbone.onnx', 'path': './models/ghost_arcface_backbone.onnx', 'expected_hash': '18bb8057d1cd3ca39411b8a4dde485fa55783e08ceecaf2352f551ca39cd1357'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.4/ghost_unet_1_block.onnx', 'path': './models/ghost_unet_1_block.onnx', 'expected_hash': '304a86bccb325e7fcf5ab4f4f84ba5172e319bccc9de15d299bb436746e2e024'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.4/ghost_unet_2_block.onnx', 'path': './models/ghost_unet_2_block.onnx', 'expected_hash': '25b72c107aabe27fc65ac5bf5377e58eda0929872d4dd3de5d5a9edefc49fa9f'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.4/ghost_unet_3_block.onnx', 'path': './models/ghost_unet_3_block.onnx', 'expected_hash': 'f471d4f322903da2bca360aa0d7ab9922e3b0001d683f825ca6b15d865382935'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.5/codeformer_fp16.onnx', 'path': './models/codeformer_fp16.onnx', 'expected_hash': '616164d3fe918fdcef68b38fd3ab5137a78ab119cc8cade25c04a1334ee3202b'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.6/VQFRv2.fp16.onnx', 'path': './models/VQFRv2.fp16.onnx', 'expected_hash': 'fd625fdc966d1e108fd84d174034b8599cc11a9da4065087cd3d67f9c7f39e8c'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.7/4x-UltraMix_Smooth.fp16.onnx', 'path': './models/4x-UltraMix_Smooth.fp16.onnx', 'expected_hash': '3b96d63c239121b1ad5992e42a2089d6b4e1185c493c6440adfeafc0a20591eb'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.7/4x-UltraSharp.fp16.onnx', 'path': './models/4x-UltraSharp.fp16.onnx', 'expected_hash': 'd801b7f6081746e0b2cccef407c7a8acdb95e284c89298684582a8f2b35ad0f9'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.7/BSRGANx2.fp16.onnx', 'path': './models/BSRGANx2.fp16.onnx', 'expected_hash': 'ba3a43613f5d2434c853201411b87e75c25ccb5b5918f38af504e4cf3bd4df9a'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.7/BSRGANx4.fp16.onnx', 'path': './models/BSRGANx4.fp16.onnx', 'expected_hash': 'e1467fbe60d2846919480f55a12ddbd5c516e343685bcdeac50ddcfa1dde2f46'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.7/ColorizeArtistic.fp16.onnx', 'path': './models/ColorizeArtistic.fp16.onnx', 'expected_hash': 'c8ad5c54b1b333361e959fdc6591828931b731f6652055f891d6118532cad081'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.7/ColorizeStable.fp16.onnx', 'path': './models/ColorizeStable.fp16.onnx', 'expected_hash': '666811485bfd37b236fdef695dbf50de7d3a430b10dbf5a3001d1609de06ad88'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.7/ColorizeVideo.fp16.onnx', 'path': './models/ColorizeVideo.fp16.onnx', 'expected_hash': '4d93b3cca8aa514bdf18a0ed00b25e36de5a9cc70b7aec7e60132632f6feced3'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.7/RealESRGAN_x2plus.fp16.onnx', 'path': './models/RealESRGAN_x2plus.fp16.onnx', 'expected_hash': '0b1770bcb31b3a9021d4251b538da4eb47c84f42706504d44a76d17e8c267606'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.7/RealESRGAN_x4plus.fp16.onnx', 'path': './models/RealESRGAN_x4plus.fp16.onnx', 'expected_hash': '0a06c68f463a14bf5563b78d77d61ba4394024e148383c4308d6d3783eac2dc5'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/2d106det.onnx', 'path': './models/2d106det.onnx', 'expected_hash': 'f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/2dfan4.onnx', 'path': './models/2dfan4.onnx', 'expected_hash': '1ceedb108439c7d7b3f92cfa2b25bdc69a1f5f6c8b41da228cb283ca98d4181d'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/79999_iter.pth', 'path': './models/79999_iter.pth', 'expected_hash': '468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/ddcolor.onnx', 'path': './models/ddcolor.onnx', 'expected_hash': '4e8b8a8d7c346ea7df08fc0bc985d30c67f5835cd1b81b6728f6bbe8b7658ae1'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/ddcolor_artistic.onnx', 'path': './models/ddcolor_artistic.onnx', 'expected_hash': '2f2510323e59995051eeac4f1ef8c267130eabf6187535defa55c11929b2b31c'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/det_10g.onnx', 'path': './models/det_10g.onnx', 'expected_hash': '5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/epoch_16_best.ckpt', 'path': './models/epoch_16_best.ckpt', 'expected_hash': '01d3c3939c28e47a45acb9a5ea8f8ee460e5ecf046c0caa8404e05915e10901b'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/face_landmarks_detector_Nx3x256x256.onnx', 'path': './models/face_landmarks_detector_Nx3x256x256.onnx', 'expected_hash': '6d7932bdefc38871f57dd915b8c723d855e599f29cf4cdf19616fb35d0ed572e'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/faceparser_fp16.onnx', 'path': './models/faceparser_fp16.onnx', 'expected_hash': '093125a0d0dde183c4ab9a53cb46499f9bc210f0c1bed5245a6448d1c63d092b'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/faceparser_resnet34.onnx', 'path': './models/faceparser_resnet34.onnx', 'expected_hash': '5b805bba7b5660ab7070b5a381dcf75e5b3e04199f1e9387232a77a00095102e'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/GFPGANv1.4.onnx', 'path': './models/GFPGANv1.4.onnx', 'expected_hash': '6548e54cbcf248af385248f0c1193b359c37a0f98b836282b09cf48af4fd2b73'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/GPEN-BFR-1024.onnx', 'path': './models/GPEN-BFR-1024.onnx', 'expected_hash': 'cec8892093d7b99828acde97bf231fb0964d3fb11b43f3b0951e36ef1e192a3e'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/GPEN-BFR-2048.onnx', 'path': './models/GPEN-BFR-2048.onnx', 'expected_hash': 'd0229ff43f979c360bd19daa9cd0ce893722d59f41a41822b9223ebbe4f89b3e'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/GPEN-BFR-256.onnx', 'path': './models/GPEN-BFR-256.onnx', 'expected_hash': 'aa5bd3ab238640a378c59e4a560f7a7150627944cf2129e6311ae4720e833271'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/GPEN-BFR-512.onnx', 'path': './models/GPEN-BFR-512.onnx', 'expected_hash': '0960f836488735444d508b588e44fb5dfd19c68fde9163ad7878aa24d1d5115e'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/grid_sample_3d_plugin.dll', 'path': './models/grid_sample_3d_plugin.dll', 'expected_hash': 'be3cb37c6ba424b163490d9f17b2a58e86f306bc0920590dfead70ecef782469'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/inswapper_128.fp16.onnx', 'path': './models/inswapper_128.fp16.onnx', 'expected_hash': '6d51a9278a1f650cffefc18ba53f38bf2769bf4bbff89267822cf72945f8a38b'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/landmark.onnx', 'path': './models/landmark.onnx', 'expected_hash': '31d22a5041326c31f19b78886939a634a5aedcaa5ab8b9b951a1167595d147db'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/meanshape_68.pkl', 'path': './models/meanshape_68.pkl', 'expected_hash': '39ffecf84ba73f0d0d7e49380833ba88713c9fcdec51df4f7ac45a48b8f4cc51'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/occluder.onnx', 'path': './models/occluder.onnx', 'expected_hash': '79f5c2edf10b83458693d122dd51488b210fb80c059c5d56347a047710d44a78'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/peppapig_teacher_Nx3x256x256.onnx', 'path': './models/peppapig_teacher_Nx3x256x256.onnx', 'expected_hash': 'd4aa6dbd0081763a6eef04bf51484175b6a133ed12999bdc83b681a03f3f87d2'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/rd64-uni-refined.pth', 'path': './models/rd64-uni-refined.pth', 'expected_hash': 'a4956f9a7978a75630b08c9d6ec075b7c51cf43b4751b686e3a011d4012ddc9d'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/realesr-general-x4v3.onnx', 'path': './models/realesr-general-x4v3.onnx', 'expected_hash': '09b757accd747d7e423c1d352b3e8f23e77cc5742d04bae958d4eb8082b76fa4'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/res50.onnx', 'path': './models/res50.onnx', 'expected_hash': '025db4efa3f7bef9911adc8eb92663608c682696a843cc7e1116d90c223354b5'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.7/RestoreFormerPlusPlus.fp16.onnx', 'path': './models/RestoreFormerPlusPlus.fp16.onnx', 'expected_hash': 'e5df99ed4f501be2009ed8e708f407dd26ac400c55a43a01d8c8c157bc475b3f'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/scrfd_2.5g_bnkps.onnx', 'path': './models/scrfd_2.5g_bnkps.onnx', 'expected_hash': 'bc24bb349491481c3ca793cf89306723162c280cb284c5a5e49df3760bf5c2ce'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/w600k_r50.onnx', 'path': './models/w600k_r50.onnx', 'expected_hash': '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/XSeg_model.onnx', 'path': './models/XSeg_model.onnx', 'expected_hash': '4381395dcbec1eef469fa71cfb381f00ac8aadc3e5decb4c29c36b6eb1f38ad9'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/yoloface_8n.onnx', 'path': './models/yoloface_8n.onnx', 'expected_hash': '2cfe93cd7db8b7326bbf971e2bdca778e053c857a8f511810b2bd833a0cdda43'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/yunet_n_640_640.onnx', 'path': './models/yunet_n_640_640.onnx', 'expected_hash': '9e65c0213faef0173a3d2e05156b4bf44a45cde598bdabb69203da4a6b7ad61e'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/appearance_feature_extractor.onnx', 'path': './models/liveportrait_onnx/appearance_feature_extractor.onnx', 'expected_hash': 'dbbbb44e4bba12302d7137bdee6a0f249b45fb6dd879509fd5baa27d70c40e32'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/motion_extractor.onnx', 'path': './models/liveportrait_onnx/motion_extractor.onnx', 'expected_hash': '99d4b3c9dd3fd301910de9415a29560e38c0afaa702da51398281376cc36fdd3'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/stitching.onnx', 'path': './models/liveportrait_onnx/stitching.onnx', 'expected_hash': '43598e9747a19f4c55d8e1604fb7d7fa70ab22377d129cb7d1fe38c9a737cc79'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/stitching_eye.onnx', 'path': './models/liveportrait_onnx/stitching_eye.onnx', 'expected_hash': '251004fe4a994c57c8cd9f2c50f3d89feb289fb42e6bc3af74470a3a9fa7d83b'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/stitching_lip.onnx', 'path': './models/liveportrait_onnx/stitching_lip.onnx', 'expected_hash': '1ca793eac4b0dc5464f1716cdaa62e595c2c2272c9971a444e39c164578dc34b'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/warping_spade-fix.onnx', 'path': './models/liveportrait_onnx/warping_spade-fix.onnx', 'expected_hash': 'a6164debbf1e851c3dcefa622111c42a78afd9bb8f1540e7d01172ddf642c3b5'}, 
    {'url': 'https://github.com/Alucard24/rope-assets/releases/download/v1.0.8/warping_spade.onnx', 'path': './models/liveportrait_onnx/warping_spade.onnx', 'expected_hash': 'd6ee9af4352b47e88e0521eba6b774c48204afddc8d91c671a5f7b8a0dfb4971'}
]

# Funzione per calcolare l'hash di un file (usiamo SHA-256 in questo esempio)
def calculate_file_hash(path, hash_algo="sha256"):
    hash_function = hashlib.new(hash_algo)
    with open(path, "rb") as file:
        while chunk := file.read(8192):
            hash_function.update(chunk)
    return hash_function.hexdigest()

# Funzione per scaricare un singolo file con barra di progresso
def download_file_with_progress(url, path):
    # Crea la directory se non esiste
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Effettua la richiesta per scaricare il file
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Verifica che la richiesta sia andata a buon fine
        total_size = int(response.headers.get('content-length', 0))  # Ottieni la dimensione totale del file
        block_size = 8192  # Dimensione del blocco di download (8 KB)
        
        # Configura la barra di progresso
        with tqdm(total=total_size, unit='iB', unit_scale=True) as progress_bar:
            with open(path, 'wb') as file:
                for chunk in response.iter_content(block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))  # Aggiorna la barra di progresso
    print(f"Download completed: {path}")

# Funzione per verificare l'integrità del file scaricato
def verify_file_integrity(path, expected_hash, hash_algo="sha256"):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return False
    calculated_hash = calculate_file_hash(path, hash_algo)
    if calculated_hash == expected_hash:
        print(f"Integrity verified: {path}")
        return True
    else:
        print(f"Integrity failed for {path}. Expected Hash: {expected_hash}, Calculated Hash: {calculated_hash}")
        return False

# Scarica tutti i file della lista e verifica l'integrità
for item in url_path_list:
    url = item["url"]
    path = item["path"]
    expected_hash = item.get("expected_hash")
    
    # Scarica il file solo se non esiste
    if not os.path.exists(path):
        download_file_with_progress(url, path)
    
    # Se c'è un hash atteso, verifica l'integrità
    if expected_hash:
        verify_file_integrity(path, expected_hash)