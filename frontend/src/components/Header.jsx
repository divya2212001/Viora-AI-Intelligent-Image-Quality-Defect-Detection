function Header({ currentPage, onNavigate }) {
  return (
    <header className="app-header">
      <div className="header-inner">

        <button
          className="brand"
          onClick={() => onNavigate("home")}
        >
          <div className="brand-icon">
            IQ
          </div>

          <div>
            <div className="brand-name">
              Viora AI
            </div>

            <div className="brand-subtitle">
              Image Quality Intelligence
            </div>
          </div>
        </button>

        <nav className="navigation">

          <button
            className={
              currentPage === "home"
                ? "nav-button active"
                : "nav-button"
            }
            onClick={() => onNavigate("home")}
          >
            Analyze
          </button>

          <button
            className={
              currentPage === "history"
                ? "nav-button active"
                : "nav-button"
            }
            onClick={() => onNavigate("history")}
          >
            History
          </button>

        </nav>

      </div>
    </header>
  );
}

export default Header;