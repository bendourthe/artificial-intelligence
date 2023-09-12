# --------------------------------------------------------------------------- #
# IMPORTS
# --------------------------------------------------------------------------- #

from unet2d import UNet as unet2d
from u2net import U2Net as u2net
from tri_unet import TriUNet as tri_unet
from gln import GLN
from gln2 import GLN2
from globalNet import GlobalNet


# --------------------------------------------------------------------------- #
# METHODS DEFINITION
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
def get_parameter_number(net):

	total_num = sum(p.numel() for p in net.parameters())
	trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

	return {'Total': total_num, 'Trainable': trainable_num}

# --------------------------------------------------------------------------- #
# JOB RUN
# --------------------------------------------------------------------------- #

#-----------------------------------------------------------------------------#
if __name__ == '__main__':

	in_channels = [1, 1, 1, 1]
	out_channels = [37, 19, 85, 6]
	globalNet_params = {'scale_factor': 0.25,
						'kernel_size': 3,
						'dilations': [1, 2, 5, 2, 1]}
	localNet_params = {'in_channels': in_channels,
					   'out_channels': out_channels}
	print('unet2d', get_parameter_number(unet2d(in_channels,
												out_channels)))
	print('u2net', get_parameter_number(u2net(in_channels,
											  out_channels)))
	print('glonet', get_parameter_number(GlobalNet(in_channels,
												   out_channels)))
	print('tri_unet2d', get_parameter_number(tri_unet(in_channels,
												      out_channels)))
	print('gu2net', get_parameter_number(GLN(u2net,
											 localNet_params,
											 globalNet_params)))
	print('g2u2net', get_parameter_number(GLN2(u2net,
											   localNet_params,
											   globalNet_params)))
