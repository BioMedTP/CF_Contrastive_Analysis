import synapseclient
import synapseutils
import os

# Initialize and authenticate with Synapse
syn = synapseclient.Synapse()
syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTczMzQ3ODA1OCwiaWF0IjoxNzMzNDc4MDU4LCJqdGkiOiIxNDQ5NyIsInN1YiI6IjM1MjQ0MDQifQ.ZatgDAO11enNE6rzrFyqTTHkKNtyxvN0wEhxor-TWuNCzV5OoIjvWIOijH99TvgJgc2wn998clVr1xXFFD3rCI_18AltS0DwhPJ-__rbX3lAYBIKZ5h6H9qy8K9NcfDzn8P_zLvA6TC7J7PlMxD7rUEISkoNpFDWK9ww29Day-MGScSZK7GnAjVmMtirwSAfp3OeIU-dVNNOUgajoqst3YrNgSpI6NgQ5v-2FmC5uuWGUPig6gNY379Zo1OVeKwyxn7OeHBEJ3FKitQrEH8ixG03i8_Me_AckX4qV3yLPPuL_vhpLraoBsYwWzRSE83NdRVxBMjsnWu2HAML_wQCVg")  # Replace "your_auth_token" with your actual token

# Define the target directory
target_dir = "/home/ids/yuhe/Shared/Data/Brain_MRI_Datasets/Tumor_BraTS2023/download"

# Sync files from Synapse
print("Downloading files...")
files = synapseutils.syncFromSynapse(syn, 'syn51156910', path=target_dir)

# # Update permissions for the downloaded files
# print("Setting permissions...")
# for root, dirs, filenames in os.walk(target_dir):
#     for dirname in dirs:
#         os.chmod(os.path.join(root, dirname), 0o775)  # Permissions for directories
#     for filename in filenames:
#         os.chmod(os.path.join(root, filename), 0o775)  # Permissions for files

# print("Download and permissions update complete!")
