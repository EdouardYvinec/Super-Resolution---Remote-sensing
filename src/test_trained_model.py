import pre_trained_model
import numpy as np
import matplotlib.pyplot as plt

def test_trained_model():
	"""
	This function runs the entire testing process by calling test_sub_pix_translation(translat = t) from pre_trained_model.py. 
	We test different values of sub-pixelic horizontal translation and gather the prediction score. These score are then plotted.

	Args : 
		- None

	Output :
		- None
	"""
	sub_pix_translat = np.linspace(0,2,20)[:-1]

	score = []
	score_tr = []
	score_tr_tr = []
	score_mutuel = []
	score_mutuel2 = []
	score_tau = []
	best_tau = []
	variances = []
	for t in sub_pix_translat:
		sc, sc_tr, sc_tr_tr, sc_mu, sc_mu2, sc_tau, b_tau, var_b_tau = pre_trained_model.test_sub_pix_translation(translat = t)
		score.append(sc)
		score_tr.append(sc_tr)
		score_tr_tr.append(sc_tr_tr)
		score_mutuel.append(sc_mu)
		score_mutuel2.append(sc_mu2)
		score_tau.append(sc_tau)
		best_tau.append(b_tau)
		variances.append(var_b_tau)
	variances = np.array(variances)
	plt.plot(sub_pix_translat, score, label = r'$\| G - P(I) \|$')
	plt.plot(sub_pix_translat, score_tr, label = r'$\| G_t - P(I_t) \|$')
	plt.plot(sub_pix_translat, score_tr_tr, label = r'$\| G - {P(I_t)}_{-t} \|$')
	plt.legend(loc="upper left")
	plt.xlabel('sub-pixel translation')
	plt.ylabel('average error')
	plt.title('evolution of the test error for sub-pixel translation\n model : pretrained')
	plt.savefig('score.png', bbox_inches = 'tight')
	plt.close()
	plt.plot(sub_pix_translat, score_mutuel, label = r'$\| P(I_t) - P(I) \|$')
	plt.plot(sub_pix_translat, score_mutuel2, label = r'$\| P(I_t) - {P(I_t)}_{-t} \|$')
	plt.legend(loc="upper left")
	plt.xlabel('sub-pixel translation')
	plt.ylabel('average error')
	plt.title('evolution of the test error for sub-pixel translation\n model : pretrained')
	plt.savefig('score2.png', bbox_inches = 'tight')
	plt.close()
	plt.plot(sub_pix_translat, score_tau, label = r'$\| {P(I_t)}_{-t} - P(I) \|$')
	plt.legend(loc="upper left")
	plt.xlabel('sub-pixel translation')
	plt.ylabel('average error')
	plt.title('evolution of the test error for sub-pixel translation\n model : pretrained')
	plt.savefig('score3.png', bbox_inches = 'tight')
	plt.close()
	plt.plot(sub_pix_translat, best_tau, label = r"$\arg\min \| {P(I_t)}_{-t'} - P(I) \|$")
	plt.fill_between(sub_pix_translat, np.array(best_tau)-variances, np.array(best_tau)+variances)
	plt.plot(sub_pix_translat, sub_pix_translat, label = r'$t$')
	plt.legend(loc="upper left")
	plt.xlabel('sub-pixel translation')
	plt.ylabel('average error')
	plt.title('best tau found for back translation on pred on translated data to match original pred\n model : pretrained')
	plt.savefig('tau.png', bbox_inches = 'tight')
	plt.close()
