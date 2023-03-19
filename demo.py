import argparse
import orignal

def parseArguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_classes', help="Number of classes", type=int, default=10)
	parser.add_argument('--n_hlayers', type=int, default=2)
	parser.add_argument('-l', '--layer_sizes', nargs='+', type=int, required=False)
	parser.add_argument('--l_rate', type=float, default=0.001)
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--optimizer', type=str, required=True)
	parser.add_argument('--activation', type=str, default='sigmoid')
	parser.add_argument('--loss', type=str, default='cross_entropy')
	parser.add_argument('--output_activation', type=str, default='softmax')
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--initializer', type=str, default='xavier')
	parser.add_argument('--hlayer_size', type=int, default=32)
	args = parser.parse_args()

	return args

if __name__ == "__main__":
	arg = parseArguments()
	
	epoch=arg.epochs
	eta=arg.l_rate
	n_layers=arg.n_hlayers
	activation_fn=arg.activation
	n_neurons=arg.hlayer_size
	initialisation=arg.initializer
	n_inputneurons=784
	n_outputneurons=10
	X=xyz.test
	Y=xyz.Y_train
	n_classes=10
	batch_size=arg.batch_size
	wgt_dec=0

	Loss,W,b=xyz.modelfit_vanila(epoch,eta,n_layers,activation_fn,n_neurons,initialisation,n_inputneurons,n_outputneurons,X,Y,n_classes,batch_size,wgt_dec)