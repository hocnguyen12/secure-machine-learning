import os
import numpy as np
import h5py
import pyhelayers
import utils
from tensorflow.keras.models import load_model

utils.verify_memory()
print('Misc. initalizations')

# Set up HE context
context = pyhelayers.DefaultContext()
print('HE context initialized')

nnp = pyhelayers.NeuralNetPlain()

# Load model from h5 and json
hyper_params = pyhelayers.PlainModelHyperParams()
nnp.init_from_files(hyper_params, ["data/cats_dogs/cats_dogs_model.json", "data/cats_dogs/cats_dogs_model.h5"])
print("loaded plain model")

# Compile model for encrypted computations
he_run_req = pyhelayers.HeRunRequirements()
he_run_req.set_he_context_options([pyhelayers.DefaultContext()])
he_run_req.optimize_for_batch_size(10)

profile = pyhelayers.HeModel.compile(nnp, he_run_req)
batch_size = profile.get_optimal_batch_size()
print('Profile ready. Batch size=', batch_size)

# Initialize HE context
context = pyhelayers.HeModel.create_context(profile)
nn = pyhelayers.NeuralNet(context)
nn.encode_encrypt(nnp, profile)
print('Encrypted network ready')

# Load test data
x_test_file = 'data/cats_dogs/x_test.h5'
y_test_file = 'data/cats_dogs/y_test.h5'
with h5py.File(x_test_file) as f:
    x_test = np.array(f["x_test"])
with h5py.File(y_test_file) as f:
    y_test = np.array(f["y_test"])

print("Loaded test data")

plain_samples, labels = utils.extract_batch(x_test, y_test, batch_size, 0)

# Encrypt data
iop = nn.create_io_processor()
samples = pyhelayers.EncryptedData(context)
iop.encode_encrypt_inputs_for_predict(samples, [plain_samples])
print("encryted test data")

utils.start_timer()

# Perform predictions
predictions = pyhelayers.EncryptedData(context)
nn.predict(predictions, samples)
print("Predicted")

duration=utils.end_timer('predict')
utils.report_duration('predict per sample',duration/batch_size)
plain_predictions = iop.decrypt_decode_output(predictions)
print('predictions',plain_predictions)

utils.assess_results(labels, plain_predictions)
