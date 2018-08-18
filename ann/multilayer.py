from numpy import array,matmul,append
from math import exp


class MultilayerPerceptron:
    def __init__(self,hidden_layer,output_layer):
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
    
    
    def activation_func(self,weight,data):
        net = matmul(data.T,weight)
        return (1-exp(-net))/(1+exp(-net))
    
    def calculate_oh(self,desired,actual):
        return (desired-actual)*(1-actual**2)*.5
    
    def train(self,output_weight,hidden_weight,pattern,label,eta,emax):
        count = 0
        while True:
            error = 0
            for j in range(len(pattern)):
                hidden_output = []
                for i in range(self.hidden_layer):
                    
                    out = self.activation_func(hidden_weight[i],pattern[j])
                    hidden_output.append(out)
                hidden_output.append(-1)
                hidden_output = array(hidden_output).reshape(self.hidden_layer+1,1)
                delta_out = array([])
                for i in range(self.output_layer):
                    out = self.activation_func(output_weight[i],hidden_output)
                    error = error + 0.5*pow((label[j]-out),2)
                    delta_out = append(delta_out,[self.calculate_oh(label[j],out)],axis=0)
                delta_out = delta_out.reshape(1,1)
                delta_hid = array([])
                for i in range(self.hidden_layer):
                    delta_hid = append(delta_hid,matmul(delta_out,output_weight.T[i])*.5*(1-hidden_output[i]**2),axis=0)
                delta_hid = delta_hid.reshape(self.hidden_layer,1)
                output_weight += eta * matmul(delta_out,hidden_output.T)
                x = pattern[j].reshape(1,len(pattern[j]))
                hidden_weight += eta * matmul(delta_hid,x)
            count += 1
            print(error)
            if error < emax:
                print(f"cycles completed: {count} \n Hidden layer weights:\n {hidden_weight}\nOutput layer weights: \n{output_weight}\n")
                return


if __name__ == "__main__":
    neuron = MultilayerPerceptron(2,1)    
    output = array([[1,2,3]],dtype=float)   
    hidden = array([[2,2,2],[-1,-1,-1]],dtype=float)   
    pattern = array([[0,0,-1],[0,1,-1],[1,0,-1],[1,1,-1]],dtype=float)   
    label = array([0,1,1,0])   
    neuron.train(output,hidden,pattern,label,.1,0.0001)