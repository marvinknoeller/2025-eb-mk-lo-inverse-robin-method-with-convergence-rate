close all
clear all
clc


load('order_delta1e-04.mat')

fig = figure();
set(gcf, 'Color', 'w')
fsize = 28;

loglog(hh,err_vec,'<-','LineWidth',3,'MarkerSize',12, 'MarkerFaceColor','w','Color','#3B5BA5')
hold on


load('order_delta1e-05.mat')
loglog(hh,err_vec,'d-','LineWidth',3,'MarkerSize',12, 'MarkerFaceColor','w','Color','#E87A5D')

load('order_delta1e-06.mat')
loglog(hh,err_vec,'o-','LineWidth',3,'MarkerSize',12, 'MarkerFaceColor','w','Color','#F3B941')

loglog(hh,hh.^2*1e3,'k--','LineWidth',3)

lgd = legend('$\sigma = 10^{-4}$','$\sigma = 10^{-5}$','$\sigma = 10^{-6}$','FontSize',fsize, 'Interpreter','Latex','LineWidth',2, ...
    'Location','SouthEast');
lgd.Position = [0.61378263745989,0.250476191157389,0.261217362540109,0.341428565979004];
lgd.Title.String = "noise level";

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

saveas(gcf,strcat('plots/','ex_3_order'),'epsc')