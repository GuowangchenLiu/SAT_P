import numpy as np



def raw_model(discharge, avg_temp, pcp_grow, GEI, a1, a2, a3, b1, b2, b3, b4 ,b5):
    A = -1/(1 + np.exp(-GEI))
    tpc = a1*discharge**b1
    # raw
    # tpc = A*a1*(discharge**(a2*avg_temp + b1)) + A*(a3*pcp_grow+b2)*(discharge**b3) + a4*discharge
    # ver1
    # tpc = A*a1*(discharge**(a2*avg_temp + b1)) + A*b3*(discharge**((a3*pcp_grow+b2))) + A*a4*discharge
    # ver2
    # tpc = A*a1*(discharge**(b1*avg_temp+b2+1)) + A*a2*(discharge**((b3*pcp_grow+b4+1))) +a3*discharge**(b5+1)
    
    #tpc[tpc<0.05] = 0.05
    return tpc

def t_model(discharge, avg_temp, pcp_grow, GEI, a1, a2, a3, b1, b2, b3, b4 ,b5):
    A = -1/(1 + np.exp(-GEI))
    # tpc = a1*discharge**b1
    # raw
    # tpc = A*a1*(discharge**(a2*avg_temp + b1)) + A*(a3*pcp_grow+b2)*(discharge**b3) + a4*discharge
    # ver1
    # tpc = A*a1*(discharge**(a2*avg_temp + b1)) + A*b3*(discharge**((a3*pcp_grow+b2))) + A*a4*discharge
    # ver2
    # tpc = A*a1*(discharge**(b1*avg_temp+b2+1)) + A*a2*(discharge**((b3*pcp_grow+b4+1))) +a3*discharge**(b5+1)
    tpc = A*a1*(discharge**(b1*avg_temp+b2+1)) 
    #tpc[tpc<0.05] = 0.05
    return tpc

def f_model(discharge, avg_temp, pcp_grow, GEI, a1, a2, a3, b1, b2, b3, b4 ,b5):
    A = -1/(1 + np.exp(-GEI))
    # tpc = a1*discharge**b1
    # raw
    # tpc = A*a1*(discharge**(a2*avg_temp + b1)) + A*(a3*pcp_grow+b2)*(discharge**b3) + a4*discharge
    # ver1
    # tpc = A*a1*(discharge**(a2*avg_temp + b1)) + A*b3*(discharge**((a3*pcp_grow+b2))) + A*a4*discharge
    # ver2
    # tpc = A*a1*(discharge**(b1*avg_temp+b2+1)) + A*a2*(discharge**((b3*pcp_grow+b4+1))) +a3*discharge**(b5+1)
    # tpc = A*a1*(discharge**(b1*pcp_grow+b2+1)) 
    tpc = A*(a1*pcp_grow+a2)*(discharge**(b1+1))
    #tpc[tpc<0.05] = 0.05
    return tpc

def tf_model(discharge, avg_temp, pcp_grow, GEI, a1, a2, a3, b1, b2, b3, b4 ,b5):
    A = -1/(1 + np.exp(-GEI))
    tpc = a1*discharge**b1
    # raw
    # tpc = A*a1*(discharge**(a2*avg_temp + b1)) + A*(a3*pcp_grow+b2)*(discharge**b3) + a4*discharge
    # ver1
    # tpc = A*a1*(discharge**(a2*avg_temp + b1)) + A*b3*(discharge**((a3*pcp_grow+b2))) + A*a4*discharge
    # ver2
    # tpc = A*a1*(discharge**(b1*avg_temp+b2+1)) + A*a2*(discharge**((b3*pcp_grow+b4+1))) +a3*discharge**(b5+1)
    tpc = a1*(discharge**(b1*avg_temp+b2+1)) + (a2*pcp_grow+a3)*(discharge**(b3+1)) +a3*discharge**(b5+1)
    
    #tpc[tpc<0.05] = 0.05
    return tpc


def SATP_model(discharge, avg_temp, pcp_grow, GEI, a1, a2, a3, b1, b2, b3, b4 ,b5):
    A = -1/(1 + np.exp(-GEI))
    # tpc = a1*discharge**b1
    # raw
    # tpc = A*a1*(discharge**(a2*avg_temp + b1)) + A*(a3*pcp_grow+b2)*(discharge**b3) + a4*discharge
    # ver1
    # tpc = A*a1*(discharge**(a2*avg_temp + b1)) + A*b3*(discharge**((a3*pcp_grow+b2))) + A*a4*discharge
    # ver2
    # tpc = A*a1*(discharge**(b1*avg_temp+b2+1)) + A*a2*(discharge**((b3*pcp_grow+b4+1))) +a3*discharge**(b5+1)
    tpc = A*a1*(discharge**(b1*avg_temp+b2+1)) + A*(a2*pcp_grow+b4)*(discharge**((b3+1))) +a3*discharge**(b5+1)
    # tpc = A*a1*(discharge**(b1*avg_temp+b2+1)) + A*(a2*pcp_grow+a3)*(discharge**(b3+1)) +a3*discharge**(b5+1)
    #tpc[tpc<0.05] = 0.05
    return tpc