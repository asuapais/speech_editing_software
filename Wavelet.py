import sys
from PyQt4 import QtCore, QtGui, QtWebKit, uic
import pywt
import IPython.display
import scipy.io.wavfile as wav
import os
import numpy as np
import IPython.display
import plotly.offline as py
import plotly
from plotly import tools
import plotly.graph_objs as go
from scipy import signal


class MyWindow(QtGui.QMainWindow,QtGui.QComboBox):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        uic.loadUi("form.ui", self)
        self.filename = ''
        self.level_decomposition=''
        self.min_AIC=[]
        self.metki_signal=[]
        self.audio=[]
        self.fn = 'A_N_A_L_Y_S_I_S.html'
        self.setWindowTitle('S I G N A L  A N A L Y S')
        self.plainTextEdit.setPlainText('')
        self.progressBar.hide()
        self.show_label.hide()
        self.pushButton.clicked.connect(self.pushButton_Click)
        self.pushButton_analiz.clicked.connect(self.pushButton_analiz_Click)
        self.graf_plot.clicked.connect(self.graf_plot_Click)
        self.show_label.clicked.connect(self.show_label_Click)
        self.comboBox_dwt.activated.connect(self.change_dwt)

    def change_dwt(self):
        if self.comboBox_dwt.currentText()=='haar':
            self.comboBox_scale.hide()
            self.label_7.hide()
        else:
            self.comboBox_scale.show()
            self.label_7.show()
    def pushButton_Click(self):
        self.progressBar.hide()
        self.progressBar.setValue(0)
        self.filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', 'C:/','*.wav')
        self.lineEdit.setText(self.filename)
        self.plainTextEdit.clear()
        return self.filename

    def pushButton_analiz_Click(self):
        self.plainTextEdit.clear()
        if self.filename=='':
            msg = QtGui.QMessageBox()
            msg.setIcon(QtGui.QMessageBox.Information)
            msg.setText("No file for analysis")
            msg.setWindowTitle("Info")
            msg.show()
            msg.exec_()
        else:
            self.progressBar.show()
            rate, self.audio = wav.read(self.filename)
            self.progressBar.setValue(10)
        # OSCILLOGRAM
            trace_audio = go.Scatter(x=range(len(self.audio)), y=self.audio, name='SIGNAL',
                                     text=range(len(self.audio)))  # marker={'size': 10, 'symbol': 'star'},mode='markers',
        # END OSCILLOGRAM

        # SETTINGS DATA
            name_dw = self.comboBox_dwt.currentText()
            self.scale_dw = int(self.comboBox_scale.currentText())
            self.level_decomposition = int(self.comboBox_level_decomposition.currentText())
            smoothing = int(self.comboBox_smoothing.currentText())
            name_cw = self.comboBox_cwt.currentText()
            #IPython.display.Audio(data=self.audio, rate=rate, url=self.fn)
        # END SETTINGS DATA
            self.progressBar.setValue(20)
        #CHARACTERISTIC FUNC
            def characteristic_func_freq(amplitude, smoothing):
                CF_freq = np.zeros([len(amplitude)])
                for k in range(len(amplitude)):
                    if k == 0:
                        CF_freq[k] = abs(amplitude[k])  # *(0.53836-0.46164*np.cos(2*np.pi*k/(len(amplitude)-1))))
                    else:
                        CF_freq[k] = abs(amplitude[k]) +4*abs((amplitude[k] - amplitude[k - 1]))  # *(0.53836-0.46164*np.cos(2*np.pi*k/(len(amplitude)-1))))
                CF_freq = signal.medfilt(CF_freq, smoothing)
                return CF_freq
        # CHARACTERISTIC FUNC

        # Calculate the Akaike Information Criteria
            def AIC_variance(amplitude):
                AIC_var = np.zeros([len(amplitude)])
                for k in range(len(amplitude)):
                    var1 = np.var(amplitude[0:k + 1])
                    var2 = np.var(amplitude[k:len(amplitude) + 1])
                    if var1 == 0 or var2 == 0:
                        AIC_var[k] = np.NaN
                    else:
                        AIC_var[k] = (k + 1) * np.log10(var1) + ((len(amplitude)) - k - 1) * np.log10(var2)
                AIC_var[0] = AIC_var[1]
                AIC_var[AIC_var.shape[0] - 1] = AIC_var[AIC_var.shape[0] - 2]
                return AIC_var
        # END Calculate the Akaike Information Criteria

        #APPROX COEF
            if name_dw=='haar':
                cA = pywt.downcoef('a', self.audio, str(name_dw), mode='symmetric',
                                   level=self.level_decomposition)
                trace_A = go.Scatter(x=range(len(cA)), y=cA, name='Approximation coefficients', text=range(len(cA)))
                y1 = characteristic_func_freq(cA, (smoothing))
                trace_characteristic_approx_freq = go.Scatter(x=range(characteristic_func_freq(cA, (smoothing)).shape[0]),
                                                              y=characteristic_func_freq(cA, (smoothing)),
                                                              name='Characteristic function Approx',
                                                              text=range(characteristic_func_freq(cA, (smoothing)).shape[0]))
            else:
                cA = pywt.downcoef('a',self.audio, str(name_dw) + str(self.scale_dw), mode='symmetric', level=(self.level_decomposition))
                trace_A = go.Scatter(x=range(len(cA)), y=cA, name='Approximation coefficients', text=range(len(cA)))
                trace_characteristic_approx_freq = go.Scatter(x=range(characteristic_func_freq(cA,(smoothing)).shape[0]),
                                                              y=characteristic_func_freq(cA,(smoothing)),
                                                              name='Characteristic function Approx',
                                                              text=range(characteristic_func_freq(cA,(smoothing)).shape[0]))
        #END APPROX COEF

            self.progressBar.setValue(30)

        #AIC APPROX
            AIC_a = np.zeros(len(cA))
            AIC_a = AIC_variance(characteristic_func_freq(cA,(smoothing)))
            trace_AIC_a = go.Scatter(x=range(len(AIC_a)), y=AIC_a, name='AIC approximation', text=range(len(AIC_a)))
        #END AIC APPROX

            self.progressBar.setValue(40)

        #CWT APPROX
            coef, freqs = pywt.cwt(cA, np.arange(1, 15), str(name_cw))
            trace_scale_a = go.Contour(colorscale='Greys', name='Scalogram approximation characteristic function',
                                       z=coef,colorbar=dict(title='Amplitude',titleside='right',thickness=25,
                                                            thicknessmode='pixels',len=0.1,lenmode='fraction', outlinewidth=0,
                                                            titlefont=dict(size=14,family='Arial, sans-serif',)))
        #END CWT APPROX

            self.progressBar.setValue(50)

        # DETAL COEF
            if name_dw=='haar':
                cD = pywt.downcoef('d', self.audio, str(name_dw), mode='symmetric',level=(self.level_decomposition))
                trace_D = go.Scatter(x=range(len(cD)), y=cD, name='Detail coefficients', text=range(len(cD)))
                y1 = characteristic_func_freq(cD, (smoothing))
                trace_characteristic_detal_freq = go.Scatter(x=range(characteristic_func_freq(cD, (smoothing)).shape[0]),
                                                             y=characteristic_func_freq(cD, (smoothing)),
                                                             name='Characteristic function Detail',
                                                             text=range(characteristic_func_freq(cD, (smoothing)).shape[0]))
            else:
                cD = pywt.downcoef('d', self.audio, str(name_dw) + str(self.scale_dw), mode='symmetric', level=(self.level_decomposition))
                trace_D = go.Scatter(x=range(len(cD)), y=cD, name='Detail coefficients', text=range(len(cD)))
                y1=characteristic_func_freq(cD, (smoothing))
                trace_characteristic_detal_freq = go.Scatter(x=range(characteristic_func_freq(cD, (smoothing)).shape[0]),
                                                             y=y1,
                                                             name='Characteristic function Detail',
                                                             text=range(characteristic_func_freq(cD, (smoothing)).shape[0]))
        # END DETAL COEF

            self.progressBar.setValue(60)

        # AIC DETAL
            AIC_d = np.zeros(len(cD))
            AIC_d = AIC_variance(characteristic_func_freq(cD, (smoothing)))
            trace_AIC_d = go.Scatter(x=range(len(AIC_d)), y=AIC_d, name='AIC detail', text=(range(len(AIC_d*(2**self.level_decomposition)))))
        # END AIC DETAL

            self.progressBar.setValue(70)

        # CWT DETAL
            coef, freqs = pywt.cwt(cD, np.arange(1, 15), str(name_cw))
            trace_scale_d = go.Contour(colorscale='Greys', showscale = False, name='Scalogram detail characteristic function',
                                       z=coef,colorbar=dict(title='Amplitude',titleside='left',thickness=25,
                                                            thicknessmode='pixels',len=0.1,))
        # END CWT DETAL

            self.progressBar.setValue(80)

            self.fig = tools.make_subplots(rows=7, cols=1, subplot_titles=('Oscillogram', 'Approximation coefficients and Characteristic function',
                                                                           'AIC approximation characteristic function','Scalogram approximation characteristic function',
                                                                           'Detail coefficients and Characteristic function','AIC detail characteristic function',
                                                                           'Scalogram detail_characteristic function'))


        #the extrema of the function
            self.min_AIC = signal.argrelmin(AIC_d[:-20], order=20)
            trace_metki_cD = go.Scatter(x=self.min_AIC[0], y=cD[self.min_AIC[0]], name='Label detail',
                                        text=(self.min_AIC[0] * (2 ** self.level_decomposition)),
                                        marker={'size': 5, 'symbol': 'circle', 'color': 'red'}, mode='markers')
            trace_metki_character = go.Scatter(x=self.min_AIC[0], y=y1[self.min_AIC[0]],
                                               name='Label_characteristic',
                                               text=(self.min_AIC[0] * (2 ** self.level_decomposition)),
                                               marker={'size': 5, 'symbol': 'circle', 'color': 'red'}, mode='markers')
            trace_metki_AIC = go.Scatter(x=self.min_AIC[0], y=AIC_d[self.min_AIC[0]], name='Label AIC detail',
                                         text=(self.min_AIC[0]*(2**self.level_decomposition)),
                                         marker={'size': 5, 'symbol': 'circle', 'color': 'red'}, mode='markers')
            self.metki_signal=self.min_AIC[0]*(2**(self.level_decomposition))
            self.plainTextEdit.clear()
            self.progressBar.setValue(90)
            self.plainTextEdit.appendPlainText('Label (number of samples):'+str(self.metki_signal))
            self.show_label.show()
        # end the extrema of the function0

            self.fig.append_trace(trace_audio, 1, 1)
            self.fig.append_trace(trace_A, 2, 1)
            self.fig.append_trace(trace_characteristic_approx_freq, 2, 1)
            self.fig.append_trace(trace_AIC_a, 3, 1)
            self.fig.append_trace(trace_scale_a, 4, 1)
            self.fig.append_trace(trace_D, 5, 1)
            self.fig.append_trace(trace_metki_cD, 5, 1)
            self.fig.append_trace(trace_characteristic_detal_freq, 5, 1)
            self.fig.append_trace(trace_metki_character, 5, 1)
            self.fig.append_trace(trace_AIC_d, 6, 1)
            self.fig.append_trace(trace_metki_AIC, 6, 1)
            self.fig.append_trace(trace_scale_d, 7, 1)

            self.fig['layout'].update(height=3000, width=1300, title='GRAPHICAL ANALYSIS')
            self.fig['layout']['xaxis1'].update(title='Coefficient')
            self.fig['layout']['xaxis2'].update(title='Coefficient')
            self.fig['layout']['xaxis3'].update(title='Coefficient')
            self.fig['layout']['xaxis4'].update(title='Coefficient')
            self.fig['layout']['xaxis5'].update(title='Coefficient')
            self.fig['layout']['xaxis6'].update(title='Coefficient')
            self.fig['layout']['xaxis7'].update(title='Coefficient')
            self.fig['layout']['yaxis1'].update(title='Amplitude')
            self.fig['layout']['yaxis2'].update(title='Amplitude')
            self.fig['layout']['yaxis3'].update(title='AIC')
            self.fig['layout']['yaxis4'].update(title='Scale')
            self.fig['layout']['yaxis5'].update(title='Amplitude')
            self.fig['layout']['yaxis6'].update(title='AIC')
            self.progressBar.setValue(100)

    def graf_plot_Click(self):
        if (self.min_AIC==[]):
            graf = QtGui.QMessageBox()
            graf.setIcon(QtGui.QMessageBox.Information)
            graf.setText("No analysis")
            graf.setWindowTitle("Attention")
            graf.show()
            graf.exec_()
        else:
            py.plot(self.fig, filename=self.fn)


    def show_label_Click(self):
        if (self.min_AIC==[]):
            show_l = QtGui.QMessageBox()
            show_l.setIcon(QtGui.QMessageBox.Information)
            show_l.setText("No analysis")
            show_l.setWindowTitle("Attention")
            show_l.show()
            show_l.exec_()
        else:
            trace_audio = go.Scatter(x=range(len(self.audio)), y=self.audio, name='SIGNAL',
                                     text=range(len(self.audio)))
            trace_audio_metki = go.Scatter(x=self.metki_signal, y=self.audio[self.metki_signal], name='Label',
                                           text=range(len(self.audio)),marker={'size': 10, 'symbol': 'star','color':'red'},mode='markers')
            layout = go.Layout(title='Oscillogram', xaxis={'title': 'Sample'}, yaxis={'title': 'Amplitude'})
            oscilogramm = go.Figure(data=[trace_audio,trace_audio_metki], layout=layout)
            py.plot(oscilogramm,filename='S_I_G_N_A_L__METKI.html')

app = QtGui.QApplication(sys.argv)
MyWindow = MyWindow()
MyWindow.show()
sys.exit(app.exec_())

