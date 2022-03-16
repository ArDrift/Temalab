namespace AdasVetelServer.model
{
    using System.Data.Linq;

    /// <summary>
    /// Defines the <see cref="AdasVetelContext" />.
    /// </summary>
    public partial class AdasVetelContext : DataContext
    {

        /// <summary>
        /// Defines the Cimek.
        /// </summary>
        public Table<Cim> Cimek;

        /// <summary>
        /// Defines the Szemelyek.
        /// </summary>
        public Table<Szemely> Szemelyek;

        /// <summary>
        /// Defines the Szervezetek.
        /// </summary>
        public Table<Szervezet> Szervezetek;

        /// <summary>
        /// Defines the Szerzodesek.
        /// </summary>
        public Table<Szerzodes> Szerzodesek;

        /// <summary>
        /// Defines the Reszvetelek.
        /// </summary>
        public Table<Reszvetel> Reszvetelek;

        /// <summary>
        /// Defines the Ingosagok.
        /// </summary>
        public Table<Ingo> Ingosagok;

        /// <summary>
        /// Defines the Ingatlanok.
        /// </summary>
        public Table<Ingatlan> Ingatlanok;

        /// <summary>
        /// Defines the Gepjarmuvek.
        /// </summary>
        public Table<Gepjarmu> Gepjarmuvek;

        /// <summary>
        /// Defines the SzerzodesTargyak.
        /// </summary>
        public Table<SzerzodesTargy> SzerzodesTargyak;

        /// <summary>
        /// Defines the Teljesitesek.
        /// </summary>
        public Table<Teljesites> Teljesitesek;

        /// <summary>
        /// Initializes a new instance of the <see cref="AdasVetelContext"/> class.
        /// </summary>
        /// <param name="connection">The connection<see cref="string"/>.</param>
        public AdasVetelContext(string connection) : base(connection)
        {

        }

    }
}
