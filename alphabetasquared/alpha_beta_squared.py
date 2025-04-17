import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, linregress, skew, kurtosis

class AlphaBetaSquared():

    # TODO:
    # - Implement self.plot_config
    # - Check for valid tables in plot_alphabeta()

    def __init__(self, *args, plot_config=None):

        self.data = self._load_data(args)

        # calculate alpha and beta, max and fir lognorm function
        self.operate_on(self.data, lambda x, *a, **kw: self.calc_alpha(x), '_alpha')
        self.operate_on(self.data, lambda x, *a, **kw: self.calc_beta(x), '_beta')
        self.operate_on(self.data, lambda x, *a, **kw: np.max(x), '_table_max')
        self.operate_on(self.data, lambda x, *a, **kw: self.calc_bimodality(x), '_bimodality')
        self.operate_on(self.data, lambda x, *a, **kw: skew(x), '_skewness')
        self.operate_on(self.data, lambda x, *a, **kw: self.calc_diptest(x), '_diptest')
        
        try:
            self.operate_on(self.data, lambda x, *a, **kw: lognorm.fit(x, floc=0), '_lognorm_fits')
        except:
            raise("[WARNING] Negative Values")
        
        # set this so that jupyternotebook/jupyterhub do not automaticly display figure
        self.auto_display = False
        self.alphabeta_scale = False
        self.save_plot = False

        # Attributes for Plot formating
        # Alpha-Beta² Plot
        self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'H', 'X', 'd']


        #TODO: Implement this actually..
        self.plot_config = {
            'color': 'blue',
            'edgecolor': 'black',
            'alpha': 0.6,
            'figsize_multiplier': 7,
            **(plot_config or {})
        }
        self.__exportable_attributes = ('_alpha', '_beta', '_size_distribution', '_lognorm_fits')
    

    def _load_data(self, args):
        """Load data from one or multiple csv files"""
        
        data_str = args[0] if len(args) == 1 and isinstance(args[0], list) else args
        data = {}
        for arg in data_str:
            if isinstance(arg, str):
                data[arg.replace('.csv', '')[arg.rfind('/')+1:]] = pd.read_csv(arg).dropna()
        return data
        

    def set_auto_display(self, x):
        self.auto_display = x

    
    def set_save_plot(self, x):
        self.save_plot = x

    
    def set_alphabeta_scale(self, x):
        self.alphabeta_scale = x

    
    def calc_alpha(self, x):
        return np.mean(np.log(x))

    
    def calc_beta(self, x):
        return np.var(np.log(x))

    def show(self):
        plt.show()

    
    def get_data(self):
        return self.data
        

    def get_alpha(self):
        return self._alpha
        

    def get_beta(self):
        return self._beta
        

    def operate_on(self, data_dict, stat_func, result_attr, *args, **kwargs):
        # Init attribute
        if not hasattr(self, result_attr):
            setattr(self, result_attr, {})
        
        # Iterate through cols of every df while creating dict based on key and writing operation result in it based on col name
        for key, tab in data_dict.items():
            result_dict = getattr(self, result_attr)
            result_dict[key] = {}
            
            for col in tab:
                result_dict[key][col] = stat_func(tab[col], key=key, *args, **kwargs)

    
    def calc_bimodality(self, data):
        n = len(data)
        numerator = skew(data)**2 + 1
        denominator = kurtosis(data) + (3 * (n - 1)**2) / ((n - 2) * (n - 3))
        return numerator / denominator

    
    def calc_diptest(self, data):
        dip, p_value = diptest.diptest(data)
        return (dip, p_value)

                                      
    def calc_distribution(self, bins=50, density=True, *args, **kwargs):
        
        def calculate_histogram(data, bins, density, *args, **kwargs):
            key = kwargs.get('key')
            if isinstance(bins, int):
                bin_edges = np.linspace(0, np.max(list(self._table_max[key].values())), bins)
            else:
                bin_edges = bins
                
            hist, _ = np.histogram(data, bins=bin_edges, density=density)
            return hist, (bin_edges[:-1] + bin_edges[1:]) / 2

        self.operate_on(self.data, lambda x, *a, **kw : calculate_histogram(x, bins, density, *a, **kw), '_size_distribution')

    
    ## Do Size Distribution in multiple methods to keep it clean.. ##    
    def plot_distribution(self, table_column_map=None):

        """Plot particle size distributions for specified tables and columns.
    
        Generates separate distribution plots for each valid table-column combination.
        Automatically calculates distributions if not already computed.
    
        Parameters
        ----------
        table_column_map : dict, optional
            Dictionary mapping table names to column lists. If None, uses all available
            columns from all tables in the computed distribution.
            Format: {table_name: [col1, col2,...]}
    
        Returns
        -------
        list of matplotlib.figure.Figure
            List of generated figure objects, one per valid table.
    
        Notes
        -----
        - Automatically skips invalid tables/columns
        - Handles display mode according to self.auto_display (important when returning figures within jupyter notebook)
    
        Examples
        --------
        >>> # Plot all available distributions
        >>> figs = obj.plot_distribution()
        
        >>> # Plot specific columns from specific tables
        >>> figs = obj.plot_distribution({
        ...     'table1': ['diameter', 'radius'],
        ...     'table2': ['size']
        ... })

        >>> # Use returned figs (for jupyter notebook self.auto_display has to be False)
        >>> figs = obj.plot_distribution()
        >>> axes = figs[0].get_axes()
        >>> axes[0].set_title("New Title")
        """
        
        if not self.auto_display:
            plt.ioff()

        if not hasattr(self, '_size_distribution'):
            self.calc_distribution()

        if table_column_map is None:
            table_column_map = {table : list(tab.keys()) for table, tab in self._size_distribution.items()}
        
        figures = [] 
        # Ensure distribution is calculated

        for key, cols in table_column_map.items():
            if not self._has_valid_table(key):
                continue
                
            cols = self._get_valid_columns(key, cols)
            if not cols:
                continue
                
            fig, axes = self._plot_table_distributions(key, cols)

            if self.save_plot:
                fig.savefig(f'{key}_size_distribution.pdf')
            
            figures.append(fig)

        if not self.auto_display:
            plt.ion()

        return figures

    
    def _validate_table_column_map(self, table_column_map):
        """Validate and normalize the table-column mapping."""
        if table_column_map is None:
            return {table: list(tab.keys()) for table, tab in self._size_distribution.items()}
        return table_column_map

    
    def _has_valid_table(self, key):
        """Check if table exists in distribution data."""
        if key not in self._size_distribution:
            print(f"[WARNING] Table '{key}' not found!")
            return False
        return True

    
    def _get_valid_columns(self, key, cols):
        """Validate and normalize columns for a table."""
        if not cols:  # If empty, use all columns
            return list(self._size_distribution[key].keys())
        return [col for col in cols if col in self._size_distribution[key]]

    
    def _plot_table_distributions(self, key, cols):
        """Create subplots for all columns in a single table."""
        n = len(cols)
        nrows, ncols = self._calculate_subplot_layout(n)
        fig, axes = self._create_figure(nrows, ncols)
        
        x_min, x_max, y_min, y_max = self._plot_all_columns(key, cols, axes)
        self._finalize_plot(fig, axes, x_min, x_max, y_min, y_max, len(cols))
        return fig, axes

    
    def _calculate_subplot_layout(self, n):
        """Calculate optimal subplot layout."""
        nrows = math.ceil(math.sqrt(n))
        ncols = math.ceil(n / nrows)
        return nrows, ncols

    
    def _create_figure(self, nrows, ncols):
        """Create figure with properly sized subplots."""
        width = ncols * 7
        height = nrows * 4
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
        return fig, axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    
    def _plot_all_columns(self, key, cols, axes):
        """Plot all columns and return min/max values."""
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        
        for i, col in enumerate(cols):
            # Get Values for bar plot
            bin_heights, bin_centers = self._size_distribution[key][col]
            bin_width = bin_centers[1] - bin_centers[0]
            #bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            #bin_width = bin_edges[1] - bin_edges[0]

            # Get values for lognorm plot
            shape, loc, scale = self._lognorm_fits[key][col]
            x_fit = np.linspace(0, self._table_max[key][col], 1000)
            
            self._plot_single_column(axes[i], bin_centers, bin_heights, bin_width, shape, loc, scale, x_fit, key, col)
            x_min, x_max, y_min, y_max = self._update_limits(bin_centers, bin_heights, x_min, x_max, y_min, y_max)
                
        return x_min, x_max, y_min, y_max
    
    def _plot_single_column(self, ax, centers, heights, width, shape, loc, scale, x_fit, title, xlabel):
        """Plot a single distribution column."""
        ax.bar(centers, heights, width=width, color="blue", edgecolor="black", alpha=0.6, label="Observed")
        ax.plot(x_fit, lognorm.pdf(x_fit, shape, loc, scale), 'r-', lw=2, label="Log-Normal Fit")
        
        # ax.text(0.01, 0.95, f'Bimodality-Coefficient: {self._bimodality[title][xlabel]:.2f}', transform=ax.transAxes)
        ax.text(0.01, 0.95, f'Skewness: {self._skewness[title][xlabel]:.2f}', transform=ax.transAxes)
        ax.text(0.01, 0.9, f'$D_n$ p-value: {self._diptest[title][xlabel][1]:.2f}', transform=ax.transAxes)
        
        ax.set_title(title + " : " + xlabel)
        ax.set_xlabel("Particle Size")
        ax.set_ylabel("Probability Density")

    
    def _update_limits(self, edges, heights, x_min, x_max, y_min, y_max):
        """Update plot limits based on current data."""
        return (
            min(x_min, np.min(edges)),
            max(x_max, np.max(edges)),
            min(y_min, np.min(heights)),
            max(y_max, np.max(heights))
        )

    
    def _finalize_plot(self, fig, axes, x_min, x_max, y_min, y_max, n_plots):
        """Apply final formatting and show plot."""
        padding_x = x_max * 0.05
        padding_y = y_max * 0.05
        
        for ax in axes[:n_plots]:
            ax.set_xlim(x_min, x_max + padding_x)
            ax.set_ylim(max(0, y_min - padding_y), y_max + padding_y)
        
        for ax in axes[n_plots:]:  # Hide unused axes
            ax.axis('off')
        
        fig.tight_layout()


    ## Do AlphaBeta² in multiple methods to keep it clean.. ##
    
    def plot_alphabeta(self, *args, **kwargs):

        """Generate an alpha-beta kinetic analysis plot.
        
        Creates a scatter plot of α vs β² values with regression lines and annotations
        indicating different kinetic regimes. Handles both automatic and manual display modes.
    
        Parameters
        ----------
        *args : tuple, optional
            Flexible column specification:
            - None: Uses all available columns from each table
            - Strings, list, dicts: Plots specified tables 
        **kwargs : dict, optional
            Additional plotting parameters (currently unused, for future extension)
    
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object and it's axes
    
        Examples
        --------
        >>> # Default plot with all available tables
        >>> obj.plot_alphabeta()
        
        >>> # Plot with specific tables
        >>> obj.plot_alphabeta("CSD_hematite_python-3", "CSD_hematite_python")
    
        >>> # Using returned figures assuming self.auto_display is false when using jupyter notebook
        >>> figs = obj.plot_alphabeta()
        >>> axes = figs.get_axes()
        >>> axes[0].set_xlim(0,5)
    
        """
        
        if not self.auto_display:
            plt.ioff()
        
        table_column_map = self._init_ab_table_column_map(args)
        x_min, x_max, y_min, y_max = self._init_ab_limits()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        x_min, x_max, y_min, y_max = self._plot_ab_data(table_column_map, x_min, x_max, y_min, y_max)
        self._plot_ab_annotations(x_max, y_max)
        self._finalize_ab_plot(x_min, x_max, y_min, y_max)

        if self.save_plot:
            plt.savefig(f'alpha_beta_plot.pdf')
        
        if not self.auto_display:
            plt.ion()

        return fig, ax

    def _init_ab_limits(self):
        """Initialize the limits of the plot based if it should be scaled or not"""
        if self.alphabeta_scale:
            return float('inf'), float('-inf'), float('inf'), float('-inf')
        return float(0), float(15), float(0), float(1.5)
        

    def _init_ab_table_column_map(self, args):
        """Initialize the table-column mapping if not provided."""
        if not args:
            return {**self._alpha}
            
        if isinstance(args[0], dict):
            return args[0]
            
        return {key: self._alpha[key] for key in args if key in self._alpha}
    

    def _plot_ab_data(self, table_column_map, x_min, x_max, y_min, y_max):
        """Iterate through table_column_map and define stuff based on amount of entries before plotting"""        
        for i, (key, tab) in enumerate(table_column_map.items()):

            color = self.color_cycle[i  % len(self.color_cycle)]
            marker = self.markers[i % len(self.markers)]
            alpha = list(self._alpha[key].values())
            beta = list(self._beta[key].values())

            ## Update the limits, if alphabeta_scale is set to True
            if self.alphabeta_scale:
                x_min, x_max, y_min, y_max = self._update_limits(alpha, beta, x_min, x_max, y_min, y_max)
                
            self._plot_ab_dataset(alpha, beta, color, marker, key)

        return x_min, x_max, y_min, y_max


    def _plot_ab_dataset(self, alpha, beta, color, marker, label):
            """Do a scatter plot of the given dataset and a linear regression"""   
            plt.scatter(alpha, beta, color = color, marker = marker, s = 30, linewidth = 0.5, label = label)

            # Linear regression with extended trend line
            factor, intercept, r_value, _, _ = linregress(alpha, beta)
            x = np.linspace(min(alpha)*0.9, max(alpha)*1.1, 200)
            y = factor * x + intercept
            plt.plot(x, y, linestyle='--', lw=2, label=f"$R²={r_value**2:.2f}$")
        

    def _plot_ab_annotations(self, x_max, y_max):
        """Plot alpha-beta region annotations."""
        specs = [
            ((0, 0.05*y_max), (0.15*x_max, 0.95*y_max), "Nucleation + Growth"),
            ((0, 0.05*y_max), (0.75*x_max, 0.85*y_max), "Surface-controlled"), 
            ((0, 0.05*y_max), (x_max, 0.05*y_max), "Supply-controlled or Random Ripening")
        ]
    
        arrow_kw = dict(arrowstyle='-|>', lw=1.5, color='black')
        text_kw = dict(fontsize=11, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none'))
    
        for (x0, y0), (x1, y1), text in specs:
            plt.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=arrow_kw)
            plt.text((x0+x1)/2, (y0+y1)/2, text, **text_kw)


    def _finalize_ab_plot(self, x_min, x_max, y_min, y_max):
        """Doing some final formatting of the figure"""
        plt.xlim(0, x_max*1.2)
        plt.ylim(0, y_max*1.2)
        plt.xlabel(r'$\alpha}$', fontsize=14)
        plt.ylabel(r'$\beta^2$', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.minorticks_on()
        plt.legend()

    ## saving data ##


    def save_data(self, attr, *args, **kwargs):
        """Saves specified attribute data to CSV files, handling complex data structures.
        
        Processes an attribute containing nested dictionary data and exports each sub-table
        as a separate CSV file. Handles both regular attributes and "private" attributes
        (with underscore prefix). Supports various data formats including scalars, arrays,
        and tuples of numpy arrays.
        
        Args
        ---------
        attr (str): Name of the attribute to save (with or without underscore prefix)
        *args: Currently unused, included for future extensibility
        **kwargs: Currently unused, included for future extensibility
        
        Note
        ---------
        - For saving multiple attributes please use obj.export_data()
        """
        # Get the attribute with or without underscore prefix
        if not hasattr(self, attr):
            if hasattr(self, '_'+attr):
                attr = '_'+attr
            else:
                raise AttributeError(f"No attribute '{attr}' found (with or without underscore)")
            
        data_dict = getattr(self, attr)
        # Make own dataframe for each table in dict        
        for tab, cols in data_dict.items():
            df = pd.DataFrame()
            # Go through entrys of the dict, chec if these are subdicts with arrays / multiple columns per subdict
            for col_name, value in cols.items():
                if(isinstance(value, tuple) and len(value) >= 2 and all(isinstance(v, np.ndarray) for v in value)):
                    for i, val in enumerate(value):
                        df[f'{col_name}_{i}'] = [val] if np.isscalar(val) else val              
                else:
                    df[col_name] = [value] if np.isscalar(value) else value
            
            df.to_csv(f'{tab}{attr}.csv', index=False)
            

    def export_data(self, *args, **kwargs):

        """Exports data either from provided arguments or from exportable attributes.
    
        This method can export data in two ways:
        1. If arguments are provided, it exports each argument by saving it.
        2. If no arguments are provided, it exports all attributes marked as exportable
           (stored in self.__exportable_attributes).
        
        Args
        -------
            *args: Variable length argument list of items to be exported.
                   Each argument will be printed and saved.
            **kwargs: Arbitrary keyword arguments (currently unused in the method,
                     but included for future extensibility).     
        Note
        -------
            This method calls self.save_data(attr)

        Example
        -------
        >>> # For specific data:
        >>> obj.export_data('alpha', 'size_distribution')

        >>> # For all exportable attributes/data
        >>> obj.export_data()
        
        """
        
        if args:
            for arg in args:
                self.save_data(arg)
        else:
            for arg in self.__exportable_attributes:
                self.save_data(arg)