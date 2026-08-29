import {
    Link,
    NavLink,
} from "react-router-dom";


function Header() {

    return (

        <header className="header">

            <div className="header-inner">

                <Link
                    to="/"
                    className="brand"
                >

                    <div className="brand-icon">
                        V
                    </div>

                    <div>
                        <div className="brand-title">
                            VIORA AI
                        </div>

                        <div className="brand-subtitle">
                            Image Quality Intelligence
                        </div>
                    </div>

                </Link>


                <nav className="navigation">

                    <NavLink
                        to="/"
                        className={({ isActive }) =>
                            isActive
                                ? "nav-link active"
                                : "nav-link"
                        }
                    >
                        Analyze
                    </NavLink>


                    <NavLink
                        to="/history"
                        className={({ isActive }) =>
                            isActive
                                ? "nav-link active"
                                : "nav-link"
                        }
                    >
                        History
                    </NavLink>

                </nav>

            </div>

        </header>
    );
}


export default Header;