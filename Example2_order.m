close all
clear all
clc


load('order_a.mat')

fig = figure();
set(gcf, 'Color', 'w')
fsize = 28;

loglog(hh,err_vec,'^b-','LineWidth',3,'MarkerSize',12, 'MarkerFaceColor','w')
hold on
loglog(hh,hh.^2*1e3,'k--','LineWidth',3)
lgd = legend('$\Vert \tilde{a} - a_h \Vert_{C^1(\partial \Omega)}$','FontSize',fsize, 'Interpreter','Latex','LineWidth',2, ...
    'Location','SouthEast');
lgd.Position = [0.502462768554688,0.25,0.377894374302455,0.089523808161418];
ax = gca;
ax.FontSize = fsize;
ax.YLim = [1e-4, 1e2];
ax.GridAlpha = .8;
ax.MinorGridAlpha = 0.2;
ax.GridLineStyle = '-';
ax.LineWidth = 2.0;
ax.TickLength = [0.02, 0.2];
ax.XLabel.String = 'h';
ax.XLabel.Interpreter = 'latex';
grid on

saveas(gcf,strcat('plots/','ex_2_order'),'epsc')