from basicsr.archs.rrdbnet_arch import RRDBNet

def get_sr_model(num_feat=64, num_block=23, num_grow_ch=32, scale=4):
    """Returns a Real-ESRGAN model."""
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_block=num_block, num_grow_ch=num_grow_ch, scale=scale)
