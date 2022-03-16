using AdasVetelServer.model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace AdasVetelServer.dbHandler
{
    public class DbHandler : IDisposable
    {
        private readonly AdasVetelContext database;


        public DbHandler()
        {
            string dbName = $"{Application.StartupPath}\\db\\AdasVetel.mdf";
            string serverName = File.ReadAllText($"{Application.StartupPath}\\config\\dbServerPath.conf");
            database = new AdasVetelContext($"Data Source={serverName};AttachDbFilename={dbName};Integrated Security=True;Connect Timeout=30");
            database.DeferredLoadingEnabled = false;
        }
        public void closeConnection()
        {
            database.Connection.Close();
        }
        public void openConnection()
        {
            database.Connection.Open();
        }

        public void deleteAllContent()
        {
            database.ExecuteCommand("DELETE FROM Részvétel");
            database.ExecuteCommand("DELETE FROM SzerződésTárgy");
            database.ExecuteCommand("DELETE FROM Szerződés");
            database.ExecuteCommand("DELETE FROM Teljesítés");
            database.ExecuteCommand("DELETE FROM Gépjármű");
            database.ExecuteCommand("DELETE FROM Ingatlan");
            database.ExecuteCommand("DELETE FROM Ingóság");
            database.ExecuteCommand("DELETE FROM Személy");
            database.ExecuteCommand("DELETE FROM Szervezet");
            database.ExecuteCommand("DELETE FROM Cím");
            database.ExecuteCommand("DELETE FROM UserData");
            // Attach the log to show generated SQL.
            database.SubmitChanges();
        }
        public void submintChanges()
        {
            database.SubmitChanges();
        }
        public void insertCimToDatabase(Cim e)
        {
            database.Cimek.InsertOnSubmit(e);
            //database.SubmitChanges();
        }
        public void insertSzervezetToDatabase(Szervezet e)
        {
            database.Szervezetek.InsertOnSubmit(e);
            // database.SubmitChanges();
        }
        public void insertSzemelyToDatabase(Szemely e)
        {
            database.Szemelyek.InsertOnSubmit(e);
            // database.SubmitChanges();
        }
        public void insertIngoToDatabase(Ingo e)
        {
            database.Ingosagok.InsertOnSubmit(e);
            // database.SubmitChanges();
        }
        public void insertIngatlanToDatabase(Ingatlan e)
        {
            database.Ingatlanok.InsertOnSubmit(e);
            // database.SubmitChanges();
        }
        public void insertSzerzodesToDatabase(Szerzodes e)
        {
            database.Szerzodesek.InsertOnSubmit(e);
            // database.SubmitChanges();
        }
        public void insertSzerzodesTargyToDatabase(SzerzodesTargy e)
        {
            database.SzerzodesTargyak.InsertOnSubmit(e);
            //  database.SubmitChanges();
        }
        public void insertReszvetelToDatabase(Reszvetel e)
        {
            database.Reszvetelek.InsertOnSubmit(e);
            // database.SubmitChanges();
        }
        public void insertGepjarmuToDatabase(Gepjarmu e)
        {
            database.Gepjarmuvek.InsertOnSubmit(e);
            // database.SubmitChanges();
        }
        public void insertTeljesitesToDatabase(Teljesites e)
        {
            database.Teljesitesek.InsertOnSubmit(e);
            // database.SubmitChanges();
        }
        public void insertUserDataToDatabase(UserData e)
        {
            database.UserData.InsertOnSubmit(e);
            // database.SubmitChanges();
        }
        public List<DbElement> getAllRecords()
        {
            List<DbElement> data = new List<DbElement>();
            data.AddRange(getCimek());
            data.AddRange(getGepjarmuvek());
            data.AddRange(getIngatlanok());
            data.AddRange(getIngosagok());
            data.AddRange(getReszvetelek());
            data.AddRange(getSzemelyek());
            data.AddRange(getSzervezetek());
            data.AddRange(getSzerzodesek());
            data.AddRange(getSzerzodesTargyak());
            data.AddRange(getTeljesitesek());
            return data;
        }

        public List<Cim> getCimek()
        {
            var Query = from tab in database.Cimek select tab;
            return Query.ToList();
        } 
        public List<Cim> getCimek(int? id)
        {
            if (id != null)
            {
                var Query = from tab in database.Cimek where tab.Az == id select tab;
                return Query.ToList();
            }
            return new List<Cim>();
        }
        public List<Cim> getCimekBySzemely(int? id)
        {
            List<Szemely> sz = getSzemelyek(id);
            if (sz.Count > 0)
                return getCimek(sz[0].LakcimAz);
            return new List<Cim>();
        }

        public List<Cim> getCimekBySzervezet(int? id)
        {
            List<Szervezet> sz = getSzervezetek(id);
            if (sz.Count > 0)
                return getCimek(sz[0].SzekhelyAz);
            return new List<Cim>();
        }
        public List<Gepjarmu> getGepjarmuvek()
        {
            var Query = from tab in database.Gepjarmuvek select tab;
            return Query.ToList();
        }
        public List<Gepjarmu> getGepjarmuvekBySzerzodes(int? id)
        {
            if (id != null)
            {
                var innerJoinQuery =
                            from szt in database.SzerzodesTargyak
                            join g in database.Gepjarmuvek on szt.GepjarmuAz equals g.Az
                            where szt.SzerzodesAz == id
                            select g;
                return innerJoinQuery.ToList();
            }
            return new List<Gepjarmu>();
        }
        public List<Ingatlan> getIngatlanok()
        {
            var Query = from tab in database.Ingatlanok select tab;
            return Query.ToList();
        }
        public List<Ingatlan> getIngatlanokBySzerzodes(int? id)
        {
            if (id != null)
            {
                var innerJoinQuery =
                            from szt in database.SzerzodesTargyak
                            join ingatlan in database.Ingatlanok on szt.IngatlanAz equals ingatlan.Az
                            where szt.SzerzodesAz == id
                            select ingatlan;
                return innerJoinQuery.ToList();
            }
            return new List<Ingatlan>();
        }
        public List<Ingo> getIngosagok()
        {
            var Query = from tab in database.Ingosagok select tab;
            return Query.ToList();
        }
        public List<Ingo> getIngosagokBySzerzodes(int? id)
        {
            if (id != null)
            {
                var innerJoinQuery =
                            from szt in database.SzerzodesTargyak
                            join ingo in database.Ingosagok on szt.IngoAz equals ingo.Az
                            where szt.SzerzodesAz == id
                            select ingo;
                return innerJoinQuery.ToList();
            }
            return new List<Ingo>();
        }
        public List<Reszvetel> getReszvetelek()
        {
            var Query = from tab in database.Reszvetelek select tab;
            return Query.ToList();
        }
        public List<Reszvetel> getReszvetelekBySzemely(int? id)
        {
            if (id != null)
            {
                var Query = from tab in database.Reszvetelek where tab.SzemelyAz == id select tab;
                return Query.ToList();
            }
            return new List<Reszvetel>();
        }
        public List<Reszvetel> getReszvetelekBySzerzodes(int? id)
        {
            if (id != null)
            {
                var Query = from tab in database.Reszvetelek where tab.SzerzodesAz == id select tab;
                return Query.ToList();
            }
            return new List<Reszvetel>();
        }
        public List<Reszvetel> getReszvetelekBySzervezet(int? id)
        {
            if (id != null)
            {
                var Query = from tab in database.Reszvetelek where tab.SzervezetAz == id select tab;
                return Query.ToList();
            }
            return new List<Reszvetel>();
        }
        public List<Szemely> getSzemelyek()
        {
            var Query = from tab in database.Szemelyek select tab;
            return Query.ToList();
        }
        public List<Szemely> getSzemelyek(int? id)
        {
            var Query = from tab in database.Szemelyek where tab.Az == id select tab;
            return Query.ToList();
        }
        public List<Szemely> getSzemelyekBySzerzodes(int? id)
        {
            if (id != null)
            {
                var innerJoinQuery =
                            from resz in database.Reszvetelek
                            join szem in database.Szemelyek on resz.SzemelyAz equals szem.Az
                            where resz.SzerzodesAz == id
                            select szem;
                return innerJoinQuery.ToList();
            }
            return new List<Szemely>();

        }
        public List<Szervezet> getSzervezetek(int? id)
        {
            var Query = from tab in database.Szervezetek where tab.Az == id select tab;
            return Query.ToList();
        }
        public List<Szervezet> getSzervezetek()
        {
            var Query = from tab in database.Szervezetek select tab;
            return Query.ToList();

        }
        public List<Szervezet> getSzervezetekBySzerzodes(int? id)
        {
            if (id != null)
            {
                var innerJoinQuery =
                            from resz in database.Reszvetelek
                            join szerv in database.Szervezetek on resz.SzervezetAz equals szerv.Az
                            where resz.SzerzodesAz == id
                            select szerv;
                return innerJoinQuery.ToList();
            }
            return new List<Szervezet>();

        }
        public List<Szerzodes> getSzerzodesekBySzemely(int? id)
        {
            if (id != null)
            {
                var innerJoinQuery =
                            from resz in database.Reszvetelek
                            join szerz in database.Szerzodesek on resz.SzerzodesAz equals szerz.Az
                            where resz.SzemelyAz == id
                            select szerz;
                return innerJoinQuery.ToList();
            }
            return new List<Szerzodes>();

        }
        public List<Szerzodes> getSzerzodesekBySzervezet(int? id)
        {
            if (id != null)
            {
                var innerJoinQuery =
                            from resz in database.Reszvetelek
                            join szerz in database.Szerzodesek on resz.SzerzodesAz equals szerz.Az
                            where resz.SzervezetAz == id
                            select szerz;
                return innerJoinQuery.ToList();
            }
            return new List<Szerzodes>();

        }
        public List<Szerzodes> getSzerzodesek()
        {
            var Query = from tab in database.Szerzodesek select tab;
            return Query.ToList();
        }
        public List<Szerzodes> getSzerzodesek(int? id)
        {
            if (id != null)
            {
                var Query = from tab in database.Szerzodesek where tab.Az == id select tab;
                return Query.ToList();
            }
            return new List<Szerzodes>();

        }
        public List<SzerzodesTargy> getSzerzodesTargyak()
        {
            var Query = from tab in database.SzerzodesTargyak select tab;
            return Query.ToList();
        }
        public List<SzerzodesTargy> getSzerzodesTargyakBySzerzodes(int? id)
        {
            if (id != null)
            {
                var Query = from tab in database.SzerzodesTargyak where tab.SzerzodesAz == id select tab;
                return Query.ToList();
            }
            return new List<SzerzodesTargy>();
        }
        public List<Teljesites> getTeljesitesek()
        {
            var Query = from tab in database.Teljesitesek select tab;
            return Query.ToList();
        }
        public List<Teljesites> getTeljesitesekBySzerzodes(int? id)
        {
            if (id != null)
            {
                var innerJoinQuery =
                            from szt in database.SzerzodesTargyak
                            join telj in database.Teljesitesek on szt.TeljesitesAz equals telj.Az
                            where szt.SzerzodesAz == id
                            select telj;
                return innerJoinQuery.ToList();
            }
            return new List<Teljesites>();
        }
        public List<UserData> getUsers(string username)
        {
            if (username != "")
            {
                var Query = from tab in database.UserData where tab.Username == username select tab;
                return Query.ToList();
            }
            return new List<UserData>();
        }


        public void Dispose()
        {
            database.Dispose();
        }
    }
}
