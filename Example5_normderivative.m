close all
clear all
clc


load('Example5_168_8delta0pz.mat')

fig = figure();
set(gcf, 'Color', 'w')
fsize = 28;

semilogy(4:16,err_vec(5:end),'^b','LineWidth',3,'MarkerSize',12, 'MarkerFaceColor','w')
lgd = legend('$\kappa(J_{F_h})$','FontSize',fsize, 'Interpreter','Latex','LineWidth',2, ...
    'Location','SouthEast');
lgd.Position = [0.633928571428571,0.212566449528649,0.251785714285713,0.132671645709447];
ax = gca;
ax.FontSize = fsize;
ax.YLim = [1e2, 1e10];
ax.GridAlpha = .8;
ax.XTick = 4:2:16;
ax.YTick = logspace(2,10,5);

ax.MinorGridAlpha = 0.2;
ax.GridLineStyle = '-';
ax.LineWidth = 2.0;
ax.TickLength = [0.02, 0.2];
ax.XLabel.String = 'J';
ax.XLabel.Interpreter = 'latex';
grid on

saveas(gcf,strcat('plots/','ex_5_condition'),'epsc')