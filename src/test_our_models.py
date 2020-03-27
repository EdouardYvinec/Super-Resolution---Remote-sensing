import Deep_model
import Deep_model_with_translat
import aliasing_test
import os
import numpy as np

def train_and_test_our_models():
	models = ['baseline', 'baseline_tanh', 'article']

	for model in models:
		if "superRes_" + model + ".ckpt.meta" not in os.listdir("../checkpoint_model/"):
			Deep_model.model(model_type = model, num_epochs = 20)
		if "superRes_" + model + "_translat.ckpt.meta" not in os.listdir("../checkpoint_model/"):
			Deep_model_with_translat.model(model_type = model, num_epochs = 20)

	sub_pix_translat = np.linspace(0,2,9)[:-1]

	for model in models:	
		scores = []
		scores_tr = []
		score = []
		score_tr = []
		score_tr_tr = []
		score_mutuel = []
		score_mutuel2 = []
		score_tau = []
		best_tau = []
		variances = []
		for t in sub_pix_translat:
			model_function = Deep_model.reload(model_type = model)
			sc, sc_tr, sc_tr_tr, sc_mu, sc_mu2, sc_tau, b_tau, b_tau_var = aliasing_test.test_fixed(
				model = model_function, 
				name_fig1 = model + 'test_aliasing_after_training.png', 
				name_fig2 = model + 'test_aliasing_after_training_diff.png',
				translat = t)
			print(t,sc, sc_tr, sc_tr_tr, sc_mu, sc_mu2, sc_tau, b_tau, b_tau_var)
			score.append(sc)
			variances.append(b_tau_var)
			score_tr.append(sc_tr)
			score_tr_tr.append(sc_tr_tr)
			score_mutuel.append(sc_mu)
			score_mutuel2.append(sc_mu2)
			score_tau.append(sc_tau)
			best_tau.append(b_tau)
		variances = np.array(variances)
		plt.plot(sub_pix_translat, score, label = r'$\| G - P(I) \|$')
		plt.plot(sub_pix_translat, score_tr, label = r'$\| G_t - P(I_t) \|$')
		plt.plot(sub_pix_translat, score_tr_tr, label = r'$\| G - {P(I_t)}_{-t} \|$')
		plt.legend(loc="upper left")
		plt.xlabel('sub-pixel translation')
		plt.ylabel('average error')
		plt.title('evolution of the test error for sub-pixel translation\n model : '+model)
		plt.savefig('score'+model+'.png', bbox_inches = 'tight')
		plt.close()
		plt.plot(sub_pix_translat, score_mutuel, label = r'$\| P(I_t) - P(I) \|$')
		plt.plot(sub_pix_translat, score_mutuel2, label = r'$\| P(I_t) - {P(I_t)}_{-t} \|$')
		plt.legend(loc="upper left")
		plt.xlabel('sub-pixel translation')
		plt.ylabel('average error')
		plt.title('evolution of the test error for sub-pixel translation\n model : '+model)
		plt.savefig('score2'+model+'.png', bbox_inches = 'tight')
		plt.close()
		plt.plot(sub_pix_translat, score_tau, label = r'$\| {P(I_t)}_{-t} - P(I) \|$')
		plt.legend(loc="upper left")
		plt.xlabel('sub-pixel translation')
		plt.ylabel('average error')
		plt.title('evolution of the test error for sub-pixel translation\n model : '+model)
		plt.savefig('score3'+model+'.png', bbox_inches = 'tight')
		plt.close()
		plt.plot(sub_pix_translat, best_tau, label = r"$\arg\min \| {P(I_t)}_{-t'} - P(I) \|$")
		plt.fill_between(sub_pix_translat, np.array(best_tau)-variances, np.array(best_tau)+variances, alpha=0.25)
		plt.plot(sub_pix_translat, sub_pix_translat, label = r'$t$')
		plt.legend(loc="upper left")
		plt.xlabel('sub-pixel translation')
		plt.ylabel('average error')
		plt.title('best tau found for back translation on pred on translated data to match original pred\n model : '+model)
		plt.savefig('tau'+model+'.png', bbox_inches = 'tight')
		plt.close()

		scores = []
		scores_tr = []
		score = []
		score_tr = []
		score_tr_tr = []
		score_mutuel = []
		score_mutuel2 = []
		score_tau = []
		best_tau = []
		variances = []
		for t in sub_pix_translat:
			model_function = Deep_model_with_translat.reload(model_type = model)
			sc, sc_tr, sc_tr_tr, sc_mu, sc_mu2, sc_tau, b_tau, b_tau_var = aliasing_test.test_fixed(
				model = model_function, 
				name_fig1 = model + 'test_aliasing_after_training.png', 
				name_fig2 = model + 'test_aliasing_after_training_diff.png',
				translat = t)
			print(t,sc, sc_tr, sc_tr_tr, sc_mu, sc_mu2, sc_tau, b_tau, b_tau_var)
			score.append(sc)
			variances.append(b_tau_var)
			score_tr.append(sc_tr)
			score_tr_tr.append(sc_tr_tr)
			score_mutuel.append(sc_mu)
			score_mutuel2.append(sc_mu2)
			score_tau.append(sc_tau)
			best_tau.append(b_tau)
		variances = np.array(variances)
		plt.plot(sub_pix_translat, score, label = r'$\| G - P(I) \|$')
		plt.plot(sub_pix_translat, score_tr, label = r'$\| G_t - P(I_t) \|$')
		plt.plot(sub_pix_translat, score_tr_tr, label = r'$\| G - {P(I_t)}_{-t} \|$')
		plt.legend(loc="upper left")
		plt.xlabel('sub-pixel translation')
		plt.ylabel('average error')
		plt.title('evolution of the test error for sub-pixel translation\n model : '+model)
		plt.savefig('score'+model+'_trans.png', bbox_inches = 'tight')
		plt.close()
		plt.plot(sub_pix_translat, score_mutuel, label = r'$\| P(I_t) - P(I) \|$')
		plt.plot(sub_pix_translat, score_mutuel2, label = r'$\| P(I_t) - {P(I_t)}_{-t} \|$')
		plt.legend(loc="upper left")
		plt.xlabel('sub-pixel translation')
		plt.ylabel('average error')
		plt.title('evolution of the test error for sub-pixel translation\n model : '+model)
		plt.savefig('score2'+model+'_trans.png', bbox_inches = 'tight')
		plt.close()
		plt.plot(sub_pix_translat, score_tau, label = r'$\| {P(I_t)}_{-t} - P(I) \|$')
		plt.legend(loc="upper left")
		plt.xlabel('sub-pixel translation')
		plt.ylabel('average error')
		plt.title('evolution of the test error for sub-pixel translation\n model : '+model)
		plt.savefig('score3'+model+'_trans.png', bbox_inches = 'tight')
		plt.close()
		plt.plot(sub_pix_translat, best_tau, label = r"$\arg\min \| {P(I_t)}_{-t'} - P(I) \|$")
		plt.fill_between(sub_pix_translat, np.array(best_tau)-variances, np.array(best_tau)+variances, alpha=0.25)
		plt.plot(sub_pix_translat, sub_pix_translat, label = r'$t$')
		plt.legend(loc="upper left")
		plt.xlabel('sub-pixel translation')
		plt.ylabel('average error')
		plt.title('best tau found for back translation on pred on translated data to match original pred\n model : '+model)
		plt.savefig('tau'+model+'_trans.png', bbox_inches = 'tight')
		plt.close()