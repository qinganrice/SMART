#########################################
#              Import Library           #
#########################################

import numpy as np
import numpy.matlib

#########################################
#            Global Parameters          #
#########################################

# Waveform params
N_OFDM_SYMS             = 24           # Number of OFDM symbols
# MOD_ORDER               = 16           # Modulation order (2/4/16/64 = BSPK/QPSK/16-QAM/64-QAM)
TX_SCALE                = 1.0          # Scale for Tdata waveform ([0:1])

# OFDM params
SC_IND_PILOTS           = np.array([7, 21, 43, 57])                           # Pilot subcarrier indices
#print(SC_IND_PILOTS)
SC_IND_DATA             = np.r_[1:7,8:21,22:27,38:43,44:57,58:64]     # Data subcarrier indices
#print(SC_IND_DATA)
N_SC                    = 64                                     # Number of subcarriers
# CP_LEN                  = 16                                    # Cyclic prefidata length
N_DATA_SYMS             = N_OFDM_SYMS * len(SC_IND_DATA)     # Number of data symbols (one per data-bearing subcarrier per OFDM symbol)

SAMP_FREQ               = 20e6

# Massive-MIMO params
# N_UE                    = 4
N_BS_ANT                = 16              # N_BS_ANT >> N_UE
# N_UPLINK_SYMBOLS        = N_OFDM_SYMS
N_0                     = 1e-2
H_var                   = 0.1


# LTS for CFO and channel estimation
lts_f = np.array([0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1])



#########################################
#      Modulation and Demodulation      #
#########################################

modvec_bpsk   =  (1/np.sqrt(2))  * np.array([-1, 1]) # and QPSK
modvec_16qam  =  (1/np.sqrt(10)) * np.array([-3, -1, +3, +1])
modvec_64qam  =  (1/np.sqrt(43)) * np.array([-7, -5, -1, -3, +7, +5, +1, +3])


def modulation (mod_order,data):
    
    if (mod_order == 2): #BPSK
        return complex(modvec_bpsk[data],0) # data = 0/1
    elif (mod_order == 4): #QPSK
        return complex(modvec_bpsk[data>>1],modvec_bpsk[np.mod(data,2)])
    elif (mod_order == 16): #16-QAM
        return complex(modvec_16qam[data>>2],modvec_16qam[np.mod(data,4)])
    elif (mod_order == 64): #64-QAM
        return complex(modvec_64qam[data>>3],modvec_64qam[np.mod(data,8)])

def demodulation (mod_order, data):

    if (mod_order == 2): #BPSK
        return float(np.real(data)>0) # data = 0/1
    elif (mod_order == 4): #QPSK
        return float(2*(np.real(data)>0) + 1*(np.imag(data)>0))
    elif (mod_order == 16): #16-QAM
        return float((8*(np.real(data)>0)) + (4*(abs(np.real(data))<0.6325)) + (2*(np.imag(data)>0)) + (1*(abs(np.imag(data))<0.6325)))
    elif (mod_order == 64): #64-QAM
        return float((32*(np.real(data)>0)) + (16*(abs(np.real(data))<0.6172)) + (8*((abs(np.real(data))<(0.9258))and((abs(np.real(data))>(0.3086))))) + (4*(np.imag(data)>0)) + (2*(abs(np.imag(data))<0.6172)) + (1*((abs(np.imag(data))<(0.9258))and((abs(np.imag(data))>(0.3086))))))

## H:(N_BS,N_UE), N_UE scalar, MOD_ORDER:(N_UE,)
def data_process (H, N_UE, MOD_ORDER):

    pilot_in_mat = np.zeros((N_UE, N_SC, N_UE))
    for i in range (0, N_UE):
        pilot_in_mat [i, :, i] = lts_f

    lts_f_mat = np.zeros((N_BS_ANT, N_SC, N_UE))
    for i in range (0, N_UE):
        lts_f_mat[:, :, i] = numpy.matlib.repmat(lts_f, N_BS_ANT, 1)

    ## Uplink

    # Generate a payload of random integers
    tx_ul_data = np.zeros((N_UE, N_DATA_SYMS),dtype='int')
    for n_ue in range (0,N_UE):
        tx_ul_data[n_ue,:] = np.random.randint(low = 0, high = MOD_ORDER[n_ue], size=(1, N_DATA_SYMS))

    # Map the data values on to complex symbols
    tx_ul_syms = np.zeros((N_UE, N_DATA_SYMS),dtype='complex')
    vec_mod = np.vectorize(modulation)
    for n_ue in range (0,N_UE):
        tx_ul_syms[n_ue,:] = vec_mod(MOD_ORDER[n_ue], tx_ul_data[n_ue,:])

    #print(tx_ul_syms.shape)

    # Reshape the symbol vector to a matrix with one column per OFDM symbol
    tx_ul_syms_mat = np.reshape(tx_ul_syms, (N_UE, len(SC_IND_DATA), N_OFDM_SYMS))

    # Define the pilot tone values as BPSK symbols
    pt_pilots = np.transpose(np.array([[1, 1, -1, 1]]))

    # Repeat the pilots across all OFDM symbols
    pt_pilots_mat = np.zeros((N_UE, 4, N_OFDM_SYMS),dtype= 'complex')

    for i in range (0,N_UE):
        pt_pilots_mat[i,:,:] = numpy.matlib.repmat(pt_pilots, 1, N_OFDM_SYMS)

    ## IFFT

    # Construct the IFFT input matrix
    data_in_mat = np.zeros((N_UE, N_SC, N_OFDM_SYMS),dtype='complex')

    # Insert the data and pilot values; other subcarriers will remain at 0
    data_in_mat[:, SC_IND_DATA, :] = tx_ul_syms_mat
    data_in_mat[:, SC_IND_PILOTS, :] = pt_pilots_mat


    tx_mat_f = np.concatenate((pilot_in_mat, data_in_mat),axis=2)
    # Reshape to a vector
    tx_payload_vec = np.reshape(tx_mat_f, (N_UE, -1))

    # UL noise matrix
    Z_mat = np.sqrt(N_0/2) * ( np.random.random((N_BS_ANT,tx_payload_vec.shape[1])) + 1j*np.random.random((N_BS_ANT,tx_payload_vec.shape[1])))

    # H = np.sqrt(H_var/2) * ( np.random.random((N_BS_ANT, N_UE)) + 1j*np.random.random((N_BS_ANT, N_UE)))
    rx_payload_vec = np.matmul(H, tx_payload_vec) + Z_mat
    rx_mat_f = np.reshape(rx_payload_vec, (N_BS_ANT, N_SC, N_UE + N_OFDM_SYMS))


    csi_mat = np.multiply(rx_mat_f[:, :, 0:N_UE], lts_f_mat)
    #print(csi_mat.shape)
    fft_out_mat = rx_mat_f[:, :, N_UE:]

    # precoding_mat = np.zeros((N_BS_ANT, N_SC, N_UE),dtype='complex')
    demult_mat = np.zeros((N_UE, N_SC, N_OFDM_SYMS),dtype='complex')
    sc_csi_mat = np.zeros((N_BS_ANT, N_UE),dtype='complex')



    for j in range (0,N_SC):
        sc_csi_mat = csi_mat[:, j, :]
        zf_mat = np.linalg.pinv(sc_csi_mat)   # ZF
        demult_mat[:, j, :] = np.matmul(zf_mat, np.squeeze(fft_out_mat[:, j, :]))



    payload_syms_mat = demult_mat[:, SC_IND_DATA, :]
    payload_syms_mat = np.reshape(payload_syms_mat, (N_UE, -1))


    tx_ul_syms_vecs = np.reshape(tx_ul_syms_mat, (N_UE, -1))
    ul_evm_mat = np.mean(np.square(np.abs(payload_syms_mat - tx_ul_syms_vecs)),1) / np.mean(np.square(np.abs(tx_ul_syms_vecs)),1)
    ul_sinrs = 1 / ul_evm_mat

    ## Spectrual Efficiency
    ul_se = np.zeros(N_UE)
    for n_ue in range (0,N_UE):
        ul_se[n_ue] = np.log2(1+ul_sinrs[n_ue])
    ul_se_total = np.sum(ul_se)


    return ul_se_total, ul_sinrs, ul_se
